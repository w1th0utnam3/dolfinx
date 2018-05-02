// Copyright (C) 2007 Magnus Vikstrøm
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include "SubSystemsManager.h"
#include <algorithm>
#include <numeric>

//-----------------------------------------------------------------------------
dolfin::MPI::Comm::Comm(MPI_Comm comm)
{
#ifdef HAS_MPI
  // Duplicate communicator
  if (comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_dup(comm, &_comm);
    if (err != MPI_SUCCESS)
      log::error("Duplication of MPI communicator failed (MPI_Comm_dup");
  }
  else
    _comm = MPI_COMM_NULL;
#else
  _comm = comm;
#endif

  std::vector<double> x = {{1.0, 3.0}};
}
//-----------------------------------------------------------------------------
dolfin::MPI::Comm::Comm(const Comm& comm) : Comm(comm._comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::MPI::Comm::Comm(Comm&& comm)
{
  this->_comm = comm._comm;
  comm._comm = MPI_COMM_NULL;
}
//-----------------------------------------------------------------------------
dolfin::MPI::Comm::~Comm() { free(); }
//-----------------------------------------------------------------------------
void dolfin::MPI::Comm::free()
{
#ifdef HAS_MPI
  if (_comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_free(&_comm);
    if (err != MPI_SUCCESS)
      std::cout << "Error when destroying communicator (MPI_Comm_free)."
                << std::endl;
  }
#endif
}
//-----------------------------------------------------------------------------
std::uint32_t dolfin::MPI::Comm::rank() const
{
  return dolfin::MPI::rank(_comm);
}
//-----------------------------------------------------------------------------
std::uint32_t dolfin::MPI::Comm::size() const
{
#ifdef HAS_MPI
  int size;
  MPI_Comm_size(_comm, &size);
  return size;
#else
  return 1;
#endif
}
//-----------------------------------------------------------------------------
void dolfin::MPI::Comm::barrier() const
{
#ifdef HAS_MPI
  MPI_Barrier(_comm);
#endif
}
//-----------------------------------------------------------------------------
void dolfin::MPI::Comm::reset(MPI_Comm comm)
{
#ifdef HAS_MPI
  if (_comm != MPI_COMM_NULL)
  {
    int err = 0;
    if (_comm != MPI_COMM_NULL)
      err = MPI_Comm_free(&_comm);

    if (err != MPI_SUCCESS)
    {
      // Raise error
    }
  }

  // Duplicate communicator
  int err = MPI_Comm_dup(comm, &_comm);
  if (err != MPI_SUCCESS)
  {
    // Raise error
  }
#else
  _comm = comm;
#endif
}
//-----------------------------------------------------------------------------
MPI_Comm dolfin::MPI::Comm::comm() const { return _comm; }
  //-----------------------------------------------------------------------------

#ifdef HAS_MPI
//-----------------------------------------------------------------------------
dolfin::MPIInfo::MPIInfo() { MPI_Info_create(&info); }
//-----------------------------------------------------------------------------
dolfin::MPIInfo::~MPIInfo() { MPI_Info_free(&info); }
//-----------------------------------------------------------------------------
MPI_Info& dolfin::MPIInfo::operator*() { return info; }
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
std::uint32_t dolfin::MPI::rank(const MPI_Comm comm)
{
#ifdef HAS_MPI
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
std::uint32_t dolfin::MPI::size(const MPI_Comm comm)
{
#ifdef HAS_MPI
  int size;
  MPI_Comm_size(comm, &size);
  return size;
#else
  return 1;
#endif
}
//-----------------------------------------------------------------------------
void dolfin::MPI::barrier(const MPI_Comm comm)
{
#ifdef HAS_MPI
  MPI_Barrier(comm);
#endif
}
//-----------------------------------------------------------------------------
std::size_t dolfin::MPI::global_offset(const MPI_Comm comm, std::size_t range,
                                       bool exclusive)
{
#ifdef HAS_MPI
  // Compute inclusive or exclusive partial reduction
  std::size_t offset = 0;
  MPI_Scan(&range, &offset, 1, mpi_type<std::size_t>(), MPI_SUM, comm);
  if (exclusive)
    offset -= range;
  return offset;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> dolfin::MPI::local_range(const MPI_Comm comm,
                                                     std::int64_t N)
{
  return local_range(comm, rank(comm), N);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2>
dolfin::MPI::local_range(const MPI_Comm comm, int process, std::int64_t N)
{
  return compute_local_range(process, N, size(comm));
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2>
dolfin::MPI::compute_local_range(int process, std::int64_t N, int size)
{
  assert(process >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (process < r)
    return {{process * (n + 1), process * (n + 1) + n + 1}};
  else
    return {{process * n + r, process * n + r + n}};
}
//-----------------------------------------------------------------------------
std::uint32_t dolfin::MPI::index_owner(const MPI_Comm comm, std::size_t index,
                                       std::size_t N)
{
  assert(index < N);

  // Get number of processes
  const std::uint32_t _size = size(comm);

  // Compute number of items per process and remainder
  const std::size_t n = N / _size;
  const std::size_t r = N % _size;

  // First r processes own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining processes own n indices
  return r + (index - r * (n + 1)) / n;
}
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
template <>
dolfin::Table dolfin::MPI::all_reduce(const MPI_Comm comm,
                                      const dolfin::Table& table,
                                      const MPI_Op op)
{
  const std::string new_title = "[" + operation_map[op] + "] " + table.name();

  // Handle trivial reduction
  if (MPI::size(comm) == 1)
  {
    Table table_all(table);
    table_all.rename(new_title);
    return table_all;
  }

  // Get keys, values into containers
  std::string keys;
  std::vector<double> values;
  keys.reserve(128 * table.dvalues.size());
  values.reserve(table.dvalues.size());
  for (auto it = table.dvalues.begin(); it != table.dvalues.end(); ++it)
  {
    keys += it->first.first + '\0' + it->first.second + '\0';
    values.push_back(it->second);
  }

  // Gather to rank zero
  std::vector<std::string> keys_all;
  std::vector<double> values_all;
  gather(comm, keys, keys_all, 0);
  gather(comm, values, values_all, 0);

  // Return empty table on rank > 0
  if (MPI::rank(comm) > 0)
    return Table(new_title);

  // Prepare reduction operation y := op(y, x)
  void (*op_impl)(double&, const double&) = NULL;
  if (op == MPI_SUM || op == MPI_AVG())
    op_impl = [](double& y, const double& x) { y += x; };
  else if (op == MPI_MIN)
    op_impl = [](double& y, const double& x) {
      if (x < y)
        y = x;
    };
  else if (op == MPI_MAX)
    op_impl = [](double& y, const double& x) {
      if (x > y)
        y = x;
    };
  else
    log::dolfin_error("MPI.h", "perform reduction of Table",
                      "MPI::reduce(comm, table, %d) not implemented", op);

  // Construct dvalues map from obtained data
  std::map<std::array<std::string, 2>, double> dvalues_all;
  std::map<std::array<std::string, 2>, double>::iterator it;
  std::array<std::string, 2> key;
  key[0].reserve(128);
  key[1].reserve(128);
  double* values_ptr = values_all.data();
  for (std::uint32_t i = 0; i != MPI::size(comm); ++i)
  {
    std::stringstream keys_stream(keys_all[i]);
    while (std::getline(keys_stream, key[0], '\0'),
           std::getline(keys_stream, key[1], '\0'))
    {
      it = dvalues_all.find(key);
      if (it != dvalues_all.end())
        op_impl(it->second, *(values_ptr++));
      else
        dvalues_all[key] = *(values_ptr++);
    }
  }
  assert(values_ptr == values_all.data() + values_all.size());

  // Weight by MPI size when averaging
  if (op == MPI_AVG())
  {
    const double w = 1.0 / static_cast<double>(size(comm));
    for (auto& it : dvalues_all)
      it.second *= w;
  }

  // Construct table to return
  Table table_all(new_title);
  for (auto& it : dvalues_all)
    table_all(it.first[0], it.first[1]) = it.second;

  return table_all;
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
template <>
dolfin::Table dolfin::MPI::avg(MPI_Comm comm, const dolfin::Table& table)
{
  return all_reduce(comm, table, MPI_AVG());
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
std::map<MPI_Op, std::string> dolfin::MPI::operation_map
    = {{MPI_SUM, "MPI_SUM"}, {MPI_MAX, "MPI_MAX"}, {MPI_MIN, "MPI_MIN"}};
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
MPI_Op dolfin::MPI::MPI_AVG()
{
  // Return dummy MPI_Op which we identify with average
  static MPI_Op op = MPI_OP_NULL;
  static MPI_User_function* fn = [](void*, void*, int*, MPI_Datatype*) {};
  if (op == MPI_OP_NULL)
  {
    MPI_Op_create(fn, 1, &op);
    operation_map[op] = "MPI_AVG";
  }
  return op;
}
#endif
//-----------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::broadcast(MPI_Comm comm, std::vector<T>& value,
                            std::uint32_t broadcaster)
{
#ifdef HAS_MPI
  // Broadcast cast size
  std::size_t bsize = value.size();
  MPI_Bcast(&bsize, 1, mpi_type<std::size_t>(), broadcaster, comm);

  // Broadcast
  value.resize(bsize);
  MPI_Bcast(const_cast<T*>(value.data()), bsize, mpi_type<T>(), broadcaster,
            comm);
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::broadcast(MPI_Comm comm, T& value, std::uint32_t broadcaster)
{
#ifdef HAS_MPI
  MPI_Bcast(&value, 1, mpi_type<T>(), broadcaster, comm);
#endif
}
//-----------------------------------------------------------------------------
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::all_to_all(MPI_Comm comm,
                             std::vector<std::vector<T>>& in_values,
                             std::vector<std::vector<T>>& out_values)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Data size per destination
  assert(in_values.size() == comm_size);
  std::vector<int> data_size_send(comm_size);
  std::vector<int> data_offset_send(comm_size + 1, 0);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_size_send[p] = in_values[p].size();
    data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
  }

  // Get received data sizes
  std::vector<int> data_size_recv(comm_size);
  MPI_Alltoall(data_size_send.data(), 1, mpi_type<int>(), data_size_recv.data(),
               1, mpi_type<int>(), comm);

  // Pack data and build receive offset
  std::vector<int> data_offset_recv(comm_size + 1, 0);
  std::vector<T> data_send(data_offset_send[comm_size]);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
    std::copy(in_values[p].begin(), in_values[p].end(),
              data_send.begin() + data_offset_send[p]);
  }

  // Send/receive data
  std::vector<T> data_recv(data_offset_recv[comm_size]);
  MPI_Alltoallv(data_send.data(), data_size_send.data(),
                data_offset_send.data(), mpi_type<T>(), data_recv.data(),
                data_size_recv.data(), data_offset_recv.data(), mpi_type<T>(),
                comm);

  // Repack data
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p].resize(data_size_recv[p]);
    std::copy(data_recv.begin() + data_offset_recv[p],
              data_recv.begin() + data_offset_recv[p + 1],
              out_values[p].begin());
  }
#else
  assert(in_values.size() == 1);
  out_values = in_values;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::all_to_all(MPI_Comm comm,
                             std::vector<std::vector<T>>& in_values,
                             std::vector<T>& out_values)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Data size per destination
  assert(in_values.size() == comm_size);
  std::vector<int> data_size_send(comm_size);
  std::vector<int> data_offset_send(comm_size + 1, 0);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_size_send[p] = in_values[p].size();
    data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
  }

  // Get received data sizes
  std::vector<int> data_size_recv(comm_size);
  MPI_Alltoall(data_size_send.data(), 1, mpi_type<int>(), data_size_recv.data(),
               1, mpi_type<int>(), comm);

  // Pack data and build receive offset
  std::vector<int> data_offset_recv(comm_size + 1, 0);
  std::vector<T> data_send(data_offset_send[comm_size]);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
    std::copy(in_values[p].begin(), in_values[p].end(),
              data_send.begin() + data_offset_send[p]);
  }

  // Send/receive data
  out_values.resize(data_offset_recv[comm_size]);
  MPI_Alltoallv(data_send.data(), data_size_send.data(),
                data_offset_send.data(), mpi_type<T>(), out_values.data(),
                data_size_recv.data(), data_offset_recv.data(), mpi_type<T>(),
                comm);

#else
  assert(in_values.size() == 1);
  out_values = in_values[0];
#endif
}
//---------------------------------------------------------------------------
// namespace dolfin
// {
// template <>
// void dolfin::MPI::all_to_all(MPI_Comm comm,
//                              std::vector<std::vector<bool>>& in_values,
//                              std::vector<std::vector<bool>>& out_values)
// {
// #ifdef HAS_MPI
//   // Copy to short int
//   std::vector<std::vector<short int>> send(in_values.size());
//   for (std::size_t i = 0; i < in_values.size(); ++i)
//     send[i].assign(in_values[i].begin(), in_values[i].end());

//   // Communicate data
//   std::vector<std::vector<short int>> recv;
//   all_to_all(comm, send, recv);

//   // Copy back to bool
//   out_values.resize(recv.size());
//   for (std::size_t i = 0; i < recv.size(); ++i)
//     out_values[i].assign(recv[i].begin(), recv[i].end());
// #else
//   assert(in_values.size() == 1);
//   out_values = in_values;
// #endif
// }
// //-----------------------------------------------------------------------------
// template <>
// void dolfin::MPI::all_to_all(MPI_Comm comm,
//                              std::vector<std::vector<bool>>& in_values,
//                              std::vector<bool>& out_values)
// {
// #ifdef HAS_MPI
//   // Copy to short int
//   std::vector<std::vector<short int>> send(in_values.size());
//   for (std::size_t i = 0; i < in_values.size(); ++i)
//     send[i].assign(in_values[i].begin(), in_values[i].end());

//   // Communicate data
//   std::vector<short int> recv;
//   all_to_all(comm, send, recv);

//   // Copy back to bool
//   out_values.assign(recv.begin(), recv.end());
// #else
//   assert(in_values.size() == 1);
//   out_values = in_values[0];
// #endif
// }
// }
// //---------------------------------------------------------------------------
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<T>& out_values,
                         std::uint32_t receiving_process)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  const int local_size = in_values.size();
  MPI_Gather(const_cast<int*>(&local_size), 1, mpi_type<int>(), pcounts.data(),
             1, mpi_type<int>(), receiving_process, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  out_values.resize(n);
  MPI_Gatherv(const_cast<T*>(in_values.data()), in_values.size(), mpi_type<T>(),
              out_values.data(), pcounts.data(), offsets.data(), mpi_type<T>(),
              receiving_process, comm);
#else
  out_values = in_values;
#endif
}
//---------------------------------------------------------------------------
void dolfin::MPI::gather(MPI_Comm comm, const std::string& in_values,
                         std::vector<std::string>& out_values,
                         std::uint32_t receiving_process)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  int local_size = in_values.size();
  MPI_Gather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT,
             receiving_process, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<char> _out(n);
  MPI_Gatherv(const_cast<char*>(in_values.data()), in_values.size(), MPI_CHAR,
              _out.data(), pcounts.data(), offsets.data(), MPI_CHAR,
              receiving_process, comm);

  // Rebuild
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p]
        = std::string(_out.begin() + offsets[p], _out.begin() + offsets[p + 1]);
  }
#else
  out_values.clear();
  out_values.push_back(in_values);
#endif
}
//-----------------------------------------------------------------------------
void dolfin::MPI::all_gather(MPI_Comm comm, const std::string& in_values,
                             std::vector<std::string>& out_values)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts(comm_size);
  int local_size = in_values.size();
  MPI_Allgather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, comm);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<char> _out(n);
  MPI_Allgatherv(const_cast<char*>(in_values.data()), in_values.size(),
                 MPI_CHAR, _out.data(), pcounts.data(), offsets.data(),
                 MPI_CHAR, comm);

  // Rebuild
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p]
        = std::string(_out.begin() + offsets[p], _out.begin() + offsets[p + 1]);
  }
#else
  out_values.clear();
  out_values.push_back(in_values);
#endif
}
//-------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                             std::vector<T>& out_values)
{
#ifdef HAS_MPI
  out_values.resize(in_values.size() * MPI::size(comm));
  MPI_Allgather(const_cast<T*>(in_values.data()), in_values.size(),
                mpi_type<T>(), out_values.data(), in_values.size(),
                mpi_type<T>(), comm);
#else
  out_values = in_values;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                             std::vector<std::vector<T>>& out_values)
{
#ifdef HAS_MPI
  const std::size_t comm_size = MPI::size(comm);

  // Get data size on each process
  std::vector<int> pcounts;
  const int local_size = in_values.size();
  MPI::all_gather(comm, local_size, pcounts);
  assert(pcounts.size() == comm_size);

  // Build offsets
  std::vector<int> offsets(comm_size + 1, 0);
  for (std::size_t i = 1; i <= comm_size; ++i)
    offsets[i] = offsets[i - 1] + pcounts[i - 1];

  // Gather data
  const std::size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
  std::vector<T> recvbuf(n);
  MPI_Allgatherv(const_cast<T*>(in_values.data()), in_values.size(),
                 mpi_type<T>(), recvbuf.data(), pcounts.data(), offsets.data(),
                 mpi_type<T>(), comm);

  // Repack data
  out_values.resize(comm_size);
  for (std::size_t p = 0; p < comm_size; ++p)
  {
    out_values[p].resize(pcounts[p]);
    for (int i = 0; i < pcounts[p]; ++i)
      out_values[p][i] = recvbuf[offsets[p] + i];
  }
#else
  out_values.clear();
  out_values.push_back(in_values);
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::all_gather(MPI_Comm comm, const T in_value,
                             std::vector<T>& out_values)
{
#ifdef HAS_MPI
  out_values.resize(MPI::size(comm));
  MPI_Allgather(const_cast<T*>(&in_value), 1, mpi_type<T>(), out_values.data(),
                1, mpi_type<T>(), comm);
#else
  out_values.clear();
  out_values.push_back(in_value);
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::scatter(MPI_Comm comm,
                          const std::vector<std::vector<T>>& in_values,
                          std::vector<T>& out_value,
                          std::uint32_t sending_process)
{
#ifdef HAS_MPI

  // Scatter number of values to each process
  const std::size_t comm_size = MPI::size(comm);
  std::vector<int> all_num_values;
  if (MPI::rank(comm) == sending_process)
  {
    assert(in_values.size() == comm_size);
    all_num_values.resize(comm_size);
    for (std::size_t i = 0; i < comm_size; ++i)
      all_num_values[i] = in_values[i].size();
  }
  int my_num_values = 0;
  scatter(comm, all_num_values, my_num_values, sending_process);

  // Prepare send buffer and offsets
  std::vector<T> sendbuf;
  std::vector<int> offsets;
  if (MPI::rank(comm) == sending_process)
  {
    // Build offsets
    offsets.resize(comm_size + 1, 0);
    for (std::size_t i = 1; i <= comm_size; ++i)
      offsets[i] = offsets[i - 1] + all_num_values[i - 1];

    // Allocate send buffer and fill
    const std::size_t n
        = std::accumulate(all_num_values.begin(), all_num_values.end(), 0);
    sendbuf.resize(n);
    for (std::size_t p = 0; p < in_values.size(); ++p)
    {
      std::copy(in_values[p].begin(), in_values[p].end(),
                sendbuf.begin() + offsets[p]);
    }
  }

  // Scatter
  out_value.resize(my_num_values);
  MPI_Scatterv(const_cast<T*>(sendbuf.data()), all_num_values.data(),
               offsets.data(), mpi_type<T>(), out_value.data(), my_num_values,
               mpi_type<T>(), sending_process, comm);
#else
  assert(sending_process == 0);
  assert(in_values.size() == 1);
  out_value = in_values[0];
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::scatter(MPI_Comm comm, const std::vector<T>& in_values,
                          T& out_value, std::uint32_t sending_process)
{
#ifdef HAS_MPI
  if (MPI::rank(comm) == sending_process)
    assert(in_values.size() == MPI::size(comm));

  MPI_Scatter(const_cast<T*>(in_values.data()), 1, mpi_type<T>(), &out_value, 1,
              mpi_type<T>(), sending_process, comm);
#else
  assert(sending_process == 0);
  assert(in_values.size() == 1);
  out_value = in_values[0];
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                            std::uint32_t dest, int send_tag,
                            std::vector<T>& recv_value, std::uint32_t source,
                            int recv_tag)
{
#ifdef HAS_MPI
  std::size_t send_size = send_value.size();
  std::size_t recv_size = 0;
  MPI_Status mpi_status;
  MPI_Sendrecv(&send_size, 1, mpi_type<std::size_t>(), dest, send_tag,
               &recv_size, 1, mpi_type<std::size_t>(), source, recv_tag, comm,
               &mpi_status);

  recv_value.resize(recv_size);
  MPI_Sendrecv(const_cast<T*>(send_value.data()), send_value.size(),
               mpi_type<T>(), dest, send_tag, recv_value.data(), recv_size,
               mpi_type<T>(), source, recv_tag, comm, &mpi_status);
#else
  log::dolfin_error("MPI.h", "call MPI::send_recv",
                    "DOLFIN has been configured without MPI support");
#endif
}
//---------------------------------------------------------------------------
template <typename T>
void dolfin::MPI::send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                            std::uint32_t dest, std::vector<T>& recv_value,
                            std::uint32_t source)
{
  MPI::send_recv(comm, send_value, dest, 0, recv_value, source, 0);
}
//---------------------------------------------------------------------------
template <typename T, typename X>
T dolfin::MPI::all_reduce(MPI_Comm comm, const T& value, X op)
{
#ifdef HAS_MPI
  T out;
  MPI_Allreduce(const_cast<T*>(&value), &out, 1, mpi_type<T>(), op, comm);
  return out;
#else
  return value;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
T dolfin::MPI::max(MPI_Comm comm, const T& value)
{
#ifdef HAS_MPI
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_MAX);
  return all_reduce(comm, value, op);
#else
  return value;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
T dolfin::MPI::min(MPI_Comm comm, const T& value)
{
#ifdef HAS_MPI
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_MIN);
  return all_reduce(comm, value, op);
#else
  return value;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
T dolfin::MPI::sum(MPI_Comm comm, const T& value)
{
#ifdef HAS_MPI
  // Enforce cast to MPI_Op; this is needed because template dispatch may
  // not recognize this is possible, e.g. C-enum to std::uint32_t in SGI MPT
  MPI_Op op = static_cast<MPI_Op>(MPI_SUM);
  return all_reduce(comm, value, op);
#else
  return value;
#endif
}
//---------------------------------------------------------------------------
template <typename T>
T dolfin::MPI::avg(MPI_Comm comm, const T& value)
{
#ifdef HAS_MPI
  log::dolfin_error("MPI.h", "perform average reduction",
                    "Not implemented for this type");
#else
  return value;
#endif
}
  //---------------------------------------------------------------------------

  // Explicit instantiations

#define ALLGATHER_MACRO(TYPE)                                                  \
  template void dolfin::MPI::all_gather<TYPE>(                                 \
      MPI_Comm, const std::vector<TYPE>&, std::vector<TYPE>&);                 \
  template void dolfin::MPI::all_gather<TYPE>(                                 \
      MPI_Comm, const std::vector<TYPE>&, std::vector<std::vector<TYPE>>&);    \
  template void dolfin::MPI::all_gather<TYPE>(MPI_Comm, const TYPE,            \
                                              std::vector<TYPE>&);
ALLGATHER_MACRO(std::size_t)
ALLGATHER_MACRO(double)
#undef ALLGATHER_MACRO

#define SCATTER_GATHER_MACRO(TYPE)                                             \
  template void dolfin::MPI::scatter<TYPE>(                                    \
      MPI_Comm, const std::vector<std::vector<TYPE>>&, std::vector<TYPE>&,     \
      std::uint32_t);                                                          \
  template void dolfin::MPI::scatter<TYPE>(MPI_Comm, const std::vector<TYPE>&, \
                                           TYPE&, std::uint32_t);              \
  template void dolfin::MPI::gather<TYPE>(MPI_Comm, const std::vector<TYPE>&,  \
                                          std::vector<TYPE>&,                  \
                                          std::uint32_t receiving_process);
SCATTER_GATHER_MACRO(std::size_t)
SCATTER_GATHER_MACRO(int)
#undef SCATTER_GATHER_MACRO

template void dolfin::MPI::broadcast<std::size_t>(MPI_Comm,
                                                  std::vector<std::size_t>&,
                                                  std::uint32_t);
template void dolfin::MPI::broadcast<std::size_t>(MPI_Comm, std::size_t&,
                                                  std::uint32_t);
template void dolfin::MPI::broadcast<int>(MPI_Comm, std::vector<int>&,
                                          std::uint32_t);

#define ALLTOALL_MACRO(TYPE)                                                   \
  template void dolfin::MPI::all_to_all<TYPE>(                                 \
      MPI_Comm, std::vector<std::vector<TYPE>>&, std::vector<TYPE>&);          \
  template void dolfin::MPI::all_to_all<TYPE>(                                 \
      MPI_Comm, std::vector<std::vector<TYPE>>&,                               \
      std::vector<std::vector<TYPE>>&);
ALLTOALL_MACRO(double)
ALLTOALL_MACRO(std::size_t)
ALLTOALL_MACRO(std::int64_t)
ALLTOALL_MACRO(std::int32_t)
ALLTOALL_MACRO(std::uint32_t)
#undef ALLTOALL_MACRO

template void
dolfin::MPI::send_recv<std::size_t>(MPI_Comm, const std::vector<std::size_t>&,
                                    std::uint32_t, std::vector<std::size_t>&,
                                    std::uint32_t);

#define REDUCTION_MACRO(TYPE)                                                  \
  template TYPE dolfin::MPI::max<TYPE>(MPI_Comm, const TYPE&);                 \
  template TYPE dolfin::MPI::min<TYPE>(MPI_Comm, const TYPE&);                 \
  template TYPE dolfin::MPI::sum<TYPE>(MPI_Comm, const TYPE&);
REDUCTION_MACRO(double)
REDUCTION_MACRO(int)
REDUCTION_MACRO(std::size_t)
REDUCTION_MACRO(std::int64_t)
#undef REDUCTION_MACRO
