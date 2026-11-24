// vision_server - keeps ExecuTorch models resident and serves inference over a Unix domain socket
// (ADR-022). Built for Android arm64 with the Qualcomm QNN (NPU) and XNNPACK (CPU) backends.
//
//   vision_server <socket path>            (launched by the Python orchestrator from the Termux app;
//                                            env: LD_LIBRARY_PATH=<npu>/lib:/vendor/lib64, ADSP_LIBRARY_PATH)
//
// Protocol: little-endian, one request -> one reply, any number of requests per connection,
// one client at a time.
//   request : u32 magic 'VSRQ' | u32 cmd | u32 name_len | name | payload
//     cmd 1 LOAD   payload: u32 path_len | path            (loads + warms the 'forward' method)
//     cmd 2 INFER  payload: u32 ndim | u32 dims[ndim] | u64 nbytes | float32 data
//     cmd 3 UNLOAD payload: -
//     cmd 4 PING   payload: -                               (name may be empty)
//     cmd 5 QUIT   payload: -
//   reply   : u32 magic 'VSRP' | i32 status (0 ok) | u32 msg_len | msg | u64 compute_us |
//             u32 n_outputs | n x (u32 ndim | u32 dims[ndim] | u64 nbytes | float32 data)
// Only float32 inputs/outputs (all our models: anomaly maps, YOLO logits/boxes).
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

constexpr uint32_t kReqMagic = 0x51525356;  // "VSRQ"
constexpr uint32_t kRepMagic = 0x50525356;  // "VSRP"
constexpr uint64_t kMaxBytes = 256ull << 20;  // refuse absurd payloads

bool read_all(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool write_all(int fd, const void* buf, size_t n) {
  const auto* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

template <typename T>
bool rd(int fd, T& v) { return read_all(fd, &v, sizeof(T)); }
template <typename T>
void put(std::vector<uint8_t>& b, T v) {
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}

bool rd_str(int fd, std::string& s) {
  uint32_t n = 0;
  if (!rd(fd, n) || n > 4096) return false;
  s.resize(n);
  return n == 0 || read_all(fd, s.data(), n);
}

struct Output {
  std::vector<uint32_t> dims;
  std::vector<float> data;
};

bool reply(int fd, int32_t status, const std::string& msg, uint64_t us, const std::vector<Output>& outs) {
  std::vector<uint8_t> b;
  put(b, kRepMagic);
  put(b, status);
  put(b, static_cast<uint32_t>(msg.size()));
  b.insert(b.end(), msg.begin(), msg.end());
  put(b, us);
  put(b, static_cast<uint32_t>(outs.size()));
  for (const auto& o : outs) {
    put(b, static_cast<uint32_t>(o.dims.size()));
    for (auto d : o.dims) put(b, d);
    put(b, static_cast<uint64_t>(o.data.size() * sizeof(float)));
    const auto* p = reinterpret_cast<const uint8_t*>(o.data.data());
    b.insert(b.end(), p, p + o.data.size() * sizeof(float));
  }
  return write_all(fd, b.data(), b.size());
}

std::map<std::string, std::unique_ptr<Module>> g_models;

uint64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Returns false if the connection should be closed.
bool handle(int fd, bool& quit) {
  uint32_t magic = 0, cmd = 0;
  if (!rd(fd, magic)) return false;
  if (magic != kReqMagic) {
    reply(fd, -1, "bad magic", 0, {});
    return false;
  }
  std::string name;
  if (!rd(fd, cmd) || !rd_str(fd, name)) return false;

  if (cmd == 4) return reply(fd, 0, "pong models=" + std::to_string(g_models.size()), 0, {});
  if (cmd == 5) {
    quit = true;
    return reply(fd, 0, "bye", 0, {});
  }
  if (cmd == 3) {
    g_models.erase(name);
    return reply(fd, 0, "unloaded " + name, 0, {});
  }
  if (cmd == 1) {
    std::string path;
    if (!rd_str(fd, path)) return false;
    const uint64_t t0 = now_us();
    auto m = std::make_unique<Module>(path, Module::LoadMode::Mmap);
    if (m->load_forward() != Error::Ok) return reply(fd, 2, "load failed: " + path, 0, {});
    g_models[name] = std::move(m);
    return reply(fd, 0, "loaded " + name, now_us() - t0, {});
  }
  if (cmd == 2) {
    uint32_t ndim = 0;
    if (!rd(fd, ndim) || ndim == 0 || ndim > 8) return false;
    std::vector<uint32_t> dims(ndim);
    if (!read_all(fd, dims.data(), ndim * sizeof(uint32_t))) return false;
    uint64_t nbytes = 0;
    if (!rd(fd, nbytes) || nbytes > kMaxBytes) return false;
    uint64_t numel = 1;
    for (auto d : dims) numel *= d;
    std::vector<float> in(nbytes / sizeof(float));
    if (!read_all(fd, in.data(), nbytes)) return false;
    if (numel * sizeof(float) != nbytes) return reply(fd, 3, "size mismatch", 0, {});
    auto it = g_models.find(name);
    if (it == g_models.end()) return reply(fd, 4, "model not loaded: " + name, 0, {});

    std::vector<executorch::aten::SizesType> sizes(dims.begin(), dims.end());
    auto tensor = from_blob(in.data(), sizes);
    const uint64_t t0 = now_us();
    auto res = it->second->forward(tensor);
    const uint64_t us = now_us() - t0;
    if (!res.ok()) return reply(fd, 5, "forward failed", us, {});

    std::vector<Output> outs;
    for (const EValue& ev : *res) {
      if (!ev.isTensor()) continue;
      const auto& t = ev.toTensor();
      if (t.scalar_type() != executorch::aten::ScalarType::Float)
        return reply(fd, 6, "non-float output", us, {});
      Output o;
      for (ssize_t i = 0; i < t.dim(); ++i) o.dims.push_back(static_cast<uint32_t>(t.size(i)));
      const float* p = t.const_data_ptr<float>();
      o.data.assign(p, p + t.numel());
      outs.push_back(std::move(o));
    }
    return reply(fd, 0, "ok", us, outs);
  }
  reply(fd, 7, "unknown command", 0, {});
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <unix socket path>\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (srv < 0) {
    std::perror("socket");
    return 1;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  ::unlink(path);
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(srv, 1) != 0) {
    std::perror("bind/listen");
    return 1;
  }
  std::printf("vision_server listening on %s\n", path);
  std::fflush(stdout);
  bool quit = false;
  while (!quit) {
    int fd = ::accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    while (!quit && handle(fd, quit)) {
    }
    ::close(fd);
  }
  ::close(srv);
  ::unlink(path);
  return 0;
}
