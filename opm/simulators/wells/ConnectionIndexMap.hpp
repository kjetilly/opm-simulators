/*
  Copyright 2016 SINTEF ICT, Applied Mathematics.
  Copyright 2016 - 2017 Statoil ASA.
  Copyright 2017 Dr. Blatt - HPC-Simulation-Software & Services
  Copyright 2016 - 2018 IRIS AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_CONNECTION_INDEX_MAP_HEADER_INCLUDED
#define OPM_CONNECTION_INDEX_MAP_HEADER_INCLUDED

#include <cstddef>
#include <vector>

namespace Opm {

/// Connection index mappings
class ConnectionIndexMap
{
public:
    /// Constructor.
    ///
    /// \param[in] numConns Total number of well connections, both open
    ///   and closed/shut.  Typically \code WellConnections::size() \endcode.
    explicit ConnectionIndexMap(const std::size_t numConns)
        : local_(numConns, -1)
    {
        this->global_.reserve(numConns);
        this->open_.reserve(numConns);
    }

    /// Enumerate/map new active connection.
    ///
    /// \param[in] connIdx Global well connection index.  Must be an
    ///   integer in the range 0..numConns-1.
    ///
    /// \param[in] connIsOpen Whether or not the connection is
    ///   open/flowing.
    void addActiveConnection(const long long  connIdx,
                             const bool connIsOpen)
    {
        this->local_[connIdx] =
            static_cast<long long>(this->global_.size());

        this->global_.push_back(connIdx);

        const auto open_conn_idx = connIsOpen
            ? this->num_open_conns_++
            : -1;

        this->open_.push_back(open_conn_idx);
    }

    /// Get local connection IDs/indices of every existing well
    /// connection.
    ///
    /// Negative value (-1) for connections that don't intersect the
    /// current rank.
    const std::vector<long long>& local() const
    {
        return this->local_;
    }

    /// Get global connection ID of local (on-rank) connection.
    ///
    /// \param[in] connIdx Local connection index.
    ///
    /// \return Global connection ID of \p connIdx.
    long long global(const long long connIdx) const
    {
        return this->global_[connIdx];
    }

    /// Get open connection ID of local (on-rank) connection.
    ///
    /// \param[in] connIdx Local connection index.
    ///
    /// \return Open connection ID of \p connIdx.  Integer in the range
    ///   0..#open connections - 1 if the connection is open or negative
    ///   value (-1) otherwise.
    long long open(const long long connIdx) const
    {
        return this->open_[connIdx];
    }

private:
    /// Local connection IDs/indices of every existing well connection.
    /// Negative value (-1) for connections that don't intersect the
    /// current rank.
    std::vector<long long> local_{};

    /// Global connection index of each on-rank reservoir connection.
    /// Reverse/transpose mapping of \c local_.
    std::vector<long long> global_{};

    /// Open connection index of each on-rank reservoir connection.
    std::vector<long long> open_{};

    /// Number of open connections on this rank.
    long long num_open_conns_{0};
};

} // namespace Opm

#endif
