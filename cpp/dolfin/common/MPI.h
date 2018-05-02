// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>
#include <cassert>

#ifdef HAS_MPI
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#endif

#include <dolfin/log/Table.h>
#include <dolfin/log/log.h>

#ifndef HAS_MPI
typedef int MPI_Comm;
#define MPI_COMM_WORLD 2
#define MPI_COMM_SELF 1
#define MPI_COMM_NULL 0
#endif

namespace dolfin
{

#ifdef HAS_MPI
class MPIInfo
{
public:
  MPIInfo();
  ~MPIInfo();
  MPI_Info& operator*();

private:
  MPI_Info info;
};
#endif

/// This class provides utility functions for easy communication
/// with MPI and handles cases when DOLFIN is not configured with
/// MPI.
class MPI
{
public:
  /// A duplicate MPI communicator and manage lifetime of the
  /// communicator
  class Comm
  {
  public:
    /// Duplicate communicator and wrap duplicate
    Comm(MPI_Comm comm);

    /// Copy constructor
    Comm(const Comm& comm);

    /// Move constructor
    Comm(Comm&& comm);

    /// Disable assignment operator
    Comm& operator=(const Comm& comm) = delete;

    /// Destructor (frees wrapped communicator)
    ~Comm();

    /// Free (destroy) communicator. Calls function 'MPI_Comm_free'.
    void free();

    /// Duplicate communivator, and free any previously created
    /// communicator
    void reset(MPI_Comm comm);

    /// Return process rank for the communicator
    std::uint32_t rank() const;

    /// Return size of the group (number of processes) associated
    /// with the communicator. This function will also intialise MPI
    /// if it hasn't already been intialised.
    std::uint32_t size() const;

    /// Set a barrier (synchronization point)
    void barrier() const;

    /// Return the underlying MPI_Comm object
    MPI_Comm comm() const;

  private:
    // MPI communicator
    MPI_Comm _comm;
  };

  /// Return process rank for the communicator
  static std::uint32_t rank(MPI_Comm comm);

  /// Return size of the group (number of processes) associated with
  /// the communicator
  static std::uint32_t size(MPI_Comm comm);

  /// Set a barrier (synchronization point)
  static void barrier(MPI_Comm comm);

  /// Send in_values[p0] to process p0 and receive values from
  /// process p1 in out_values[p1]
  template <typename T>
  static void all_to_all(MPI_Comm comm, std::vector<std::vector<T>>& in_values,
                         std::vector<std::vector<T>>& out_values);

  /// Send in_values[p0] to process p0 and receive values from
  /// all processes in out_values
  template <typename T>
  static void all_to_all(MPI_Comm comm, std::vector<std::vector<T>>& in_values,
                         std::vector<T>& out_values);

  /// Broadcast vector of value from broadcaster to all processes
  template <typename T>
  static void broadcast(MPI_Comm comm, std::vector<T>& value,
                        std::uint32_t broadcaster = 0);

  /// Broadcast single primitive from broadcaster to all processes
  template <typename T>
  static void broadcast(MPI_Comm comm, T& value, std::uint32_t broadcaster = 0);

  /// Scatter vector in_values[i] to process i
  template <typename T>
  static void
  scatter(MPI_Comm comm, const std::vector<std::vector<T>>& in_values,
          std::vector<T>& out_value, std::uint32_t sending_process = 0);

  /// Scatter primitive in_values[i] to process i
  template <typename T>
  static void scatter(MPI_Comm comm, const std::vector<T>& in_values,
                      T& out_value, std::uint32_t sending_process = 0);

  /// Gather values on one process
  template <typename T>
  static void gather(MPI_Comm comm, const std::vector<T>& in_values,
                     std::vector<T>& out_values,
                     std::uint32_t receiving_process = 0);

  /// Gather strings on one process
  static void gather(MPI_Comm comm, const std::string& in_values,
                     std::vector<std::string>& out_values,
                     std::uint32_t receiving_process = 0);

  /// Gather values from all processes. Same data count from each
  /// process (wrapper for MPI_Allgather)
  template <typename T>
  static void all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<T>& out_values);

  /// Gather values from each process (variable count per process)
  template <typename T>
  static void all_gather(MPI_Comm comm, const std::vector<T>& in_values,
                         std::vector<std::vector<T>>& out_values);

  /// Gather values, one primitive from each process (MPI_Allgather)
  template <typename T>
  static void all_gather(MPI_Comm comm, const T in_value,
                         std::vector<T>& out_values);

  /// Gather values, one primitive from each process (MPI_Allgather).
  /// Specialization for std::string
  static void all_gather(MPI_Comm comm, const std::string& in_values,
                         std::vector<std::string>& out_values);

  /// Return global max value
  template <typename T>
  static T max(MPI_Comm comm, const T& value);

  /// Return global min value
  template <typename T>
  static T min(MPI_Comm comm, const T& value);

  /// Sum values and return sum
  template <typename T>
  static T sum(MPI_Comm comm, const T& value);

  /// Return average across comm; implemented only for T == Table
  template <typename T>
  static T avg(MPI_Comm comm, const T& value);

  /// All reduce
  template <typename T, typename X>
  static T all_reduce(MPI_Comm comm, const T& value, X op);

  /// Find global offset (index) (wrapper for MPI_(Ex)Scan with
  /// MPI_SUM as reduction op)
  static std::size_t global_offset(MPI_Comm comm, std::size_t range,
                                   bool exclusive);

  /// Send-receive data between processes (blocking)
  template <typename T>
  static void send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                        std::uint32_t dest, int send_tag,
                        std::vector<T>& recv_value, std::uint32_t source,
                        int recv_tag);

  /// Send-receive data between processes
  template <typename T>
  static void send_recv(MPI_Comm comm, const std::vector<T>& send_value,
                        std::uint32_t dest, std::vector<T>& recv_value,
                        std::uint32_t source);

  /// Return local range for local process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2> local_range(MPI_Comm comm, std::int64_t N);

  /// Return local range for given process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2> local_range(MPI_Comm comm, int process,
                                                 std::int64_t N);

  /// Return local range for given process, splitting [0, N - 1] into
  /// size() portions of almost equal size
  static std::array<std::int64_t, 2>
  compute_local_range(int process, std::int64_t N, int size);

  /// Return which process owns index (inverse of local_range)
  static std::uint32_t index_owner(MPI_Comm comm, std::size_t index,
                                   std::size_t N);

#ifdef HAS_MPI
  /// Return average reduction operation; recognized by
  /// all_reduce(MPI_Comm, Table&, MPI_Op)
  static MPI_Op MPI_AVG();
#endif

private:
#ifndef HAS_MPI
  static void error_no_mpi(const char* where)
  {
    log::dolfin_error("MPI.h", where,
                 "DOLFIN has been configured without MPI support");
  }
#endif

#ifdef HAS_MPI
  // Return MPI data type
  template <typename T>
  struct dependent_false : std::false_type
  {
  };
  template <typename T>
  static MPI_Datatype mpi_type()
  {
    static_assert(dependent_false<T>::value, "Unknown MPI type");
    log::dolfin_error("MPI.h", "perform MPI operation", "MPI data type unknown");
    return MPI_CHAR;
  }
#endif

#ifdef HAS_MPI
  // Maps some MPI_Op values to string
  static std::map<MPI_Op, std::string> operation_map;
#endif
};

#ifdef HAS_MPI
// Specialisations for MPI_Datatypes
template <>
inline MPI_Datatype MPI::mpi_type<float>()
{
  return MPI_FLOAT;
}
template <>
inline MPI_Datatype MPI::mpi_type<double>()
{
  return MPI_DOUBLE;
}
template <>
inline MPI_Datatype MPI::mpi_type<short int>()
{
  return MPI_SHORT;
}
template <>
inline MPI_Datatype MPI::mpi_type<int>()
{
  return MPI_INT;
}
template <>
inline MPI_Datatype MPI::mpi_type<long int>()
{
  return MPI_LONG;
}
template <>
inline MPI_Datatype MPI::mpi_type<std::uint32_t>()
{
  return MPI_UNSIGNED;
}
template <>
inline MPI_Datatype MPI::mpi_type<unsigned long int>()
{
  return MPI_UNSIGNED_LONG;
}
template <>
inline MPI_Datatype MPI::mpi_type<long long>()
{
  return MPI_LONG_LONG;
}
#endif
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
// Specialization for dolfin::log::Table class
// NOTE: This function is not trully "all_reduce", it reduces to rank 0
//       and returns zero Table on other ranks.
#ifdef HAS_MPI
template <>
Table dolfin::MPI::all_reduce(MPI_Comm, const Table&, MPI_Op);
//---------------------------------------------------------------------------
// Specialization for dolfin::log::Table class
template <>
Table dolfin::MPI::avg(MPI_Comm, const Table&);
#endif
//---------------------------------------------------------------------------
}
