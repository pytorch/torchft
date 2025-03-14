#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "ProcessGroupNCCL.hpp"

namespace py = pybind11;

namespace {
// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(
      IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      // NOLINTNEXTLINE(bugprone-exception-escape)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

PYBIND11_MODULE(_torchft_cpp, module) {
  py::object backend =
      (py::object)py::module_::import("torch._C._distributed_c10d")
          .attr("Backend");

  auto processGroupNCCL =
      intrusive_ptr_no_gil_destructor_class_<::torchft::ProcessGroupNCCL>(
          module, "ProcessGroupNCCL", backend)
          .def(
              py::init(
                  [](const c10::intrusive_ptr<::c10d::Store>& store,
                     int rank,
                     int size,
                     c10::intrusive_ptr<::torchft::ProcessGroupNCCL::Options>
                         options) {
                    // gil_scoped_release is not safe as a call_guard in init.
                    // https://github.com/pybind/pybind11/issues/5473
                    py::gil_scoped_release nogil{};

                    return c10::make_intrusive<::torchft::ProcessGroupNCCL>(
                        store, rank, size, std::move(options));
                  }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("options"),
              R"(Create a new ProcessGroupNCCL instance.)")
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                auto options = ::torchft::ProcessGroupNCCL::Options::create();
                options->is_high_priority_stream = false;
                options->timeout = timeout;
                return c10::make_intrusive<::torchft::ProcessGroupNCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = ::torchft::kProcessGroupNCCLDefaultTimeout,
              R"(Create a new ProcessGroupNCCL instance.)")
          .def("_group_start", &::torchft::ProcessGroupNCCL::groupStart)
          .def("_group_end", &::torchft::ProcessGroupNCCL::groupEnd)
          .def(
              "comm_split_count",
              &::torchft::ProcessGroupNCCL::getCommSplitCounter)
          .def(
              "_set_default_timeout",
              [](const c10::intrusive_ptr<::torchft::ProcessGroupNCCL>& self,
                 std::chrono::milliseconds timeout) {
                self->getOptions()->timeout = timeout;
              },
              py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_add_ephemeral_timeout",
              [](const c10::intrusive_ptr<::torchft::ProcessGroupNCCL>& self,
                 const std::chrono::milliseconds& timeout) {
                self->addEphemeralTimeout(timeout);
              },
              py::arg("timeout"))
          .def(
              "_verify_work_timeout",
              [](const c10::intrusive_ptr<::torchft::ProcessGroupNCCL>& self,
                 const c10::intrusive_ptr<::c10d::Work>& work,
                 const std::chrono::milliseconds& timeout) {
                return self->verifyWorkTimeoutForTest(work, timeout);
              },
              py::arg("work"),
              py::arg("timeout"))
          .def_property_readonly(
              "options",
              &::torchft::ProcessGroupNCCL::getOptions,
              R"(Return the options used to create this ProcessGroupNCCL instance.)")
          .def_property_readonly(
              "uid", &::torchft::ProcessGroupNCCL::getUid, R"(Return the uid.)")
          .def_property(
              "bound_device_id",
              &::torchft::ProcessGroupNCCL::getBoundDeviceId,
              &::torchft::ProcessGroupNCCL::setBoundDeviceId,
              R"(Return the bound device id.)")
          .def(
              "perform_nocolor_split",
              &::torchft::ProcessGroupNCCL::performNocolorSplit)
          .def(
              "register_mem_pool",
              &::torchft::ProcessGroupNCCL::registerMemPool)
          .def(
              "deregister_mem_pool",
              &::torchft::ProcessGroupNCCL::deregisterMemPool)
          .def(
              "_is_initialized",
              &::torchft::ProcessGroupNCCL::isInitialized,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_error",
              &::torchft::ProcessGroupNCCL::getError,
              py::call_guard<py::gil_scoped_release>());
}
