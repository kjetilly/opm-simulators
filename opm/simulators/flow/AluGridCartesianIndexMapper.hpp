// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/

#ifndef OPM_ALUGRID_CARTESIAN_INDEX_MAPPER_HPP
#define OPM_ALUGRID_CARTESIAN_INDEX_MAPPER_HPP

#include <dune/alugrid/grid.hh>
#include <dune/alugrid/3d/gridview.hh>
#include <opm/grid/common/CartesianIndexMapper.hpp>
#include <dune/grid/common/datahandleif.hh>
#include <dune/grid/utility/persistentcontainer.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Dune {

/*!
 * \brief Interface class to access the logical Cartesian grid as used in industry
 *        standard simulator decks.
 */
//#if HAVE_MPI
    template <>
    class CartesianIndexMapper<Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming>>
{
public:

#if HAVE_MPI
    using Grid = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming, Dune::ALUGridMPIComm>; 
#else    
    using Grid = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming, Dune::ALUGridNoComm>;
#endif //HAVE_MPI
    
    // data handle for communicating global ids during load balance and communication
    template <class GridView>
    class GlobalIndexDataHandle : public Dune::CommDataHandleIF<GlobalIndexDataHandle<GridView>, long long>
    {
        // global id
        class GlobalCellIndex
        {
        public:
            GlobalCellIndex()
                : idx_(-1)
            {}

            GlobalCellIndex& operator=(const long long index)
            {
                idx_ = index;
                return *this;
            }

            long long index() const
            { return idx_; }

        private:
            long long idx_;
        };

        using GlobalIndexContainer = typename Dune::PersistentContainer<Grid, GlobalCellIndex>;

    public:
        // constructor copying cartesian index to persistent container
        GlobalIndexDataHandle(const GridView& gridView,
                               std::vector<long long>& cartesianIndex)
            : gridView_(gridView)
            , globalIndex_(gridView.grid(), 0)
            , cartesianIndex_(cartesianIndex)
        {
            globalIndex_.resize();
            initialize();
        }

        // constructor copying cartesian index to persistent container
        GlobalIndexDataHandle(const GlobalIndexDataHandle& other) = delete ;

        // destrcutor writing load balanced cartesian index back to vector
        ~GlobalIndexDataHandle()
        { finalize(); }

        bool contains(long long /* dim */, long long codim) const
        { return codim == 0; }

        bool fixedsize(long long /* dim */, long long /* codim */) const
        { return true; }

        //! \brief loop over all internal data handlers and call gather for
        //! given entity
        template<class MessageBufferImp, class EntityType>
        void gather(MessageBufferImp& buff, const EntityType& element) const
        {
            long long globalIdx = globalIndex_[element].index();
            buff.write(globalIdx);
        }

        //! \brief loop over all internal data handlers and call scatter for
        //! given entity
        template<class MessageBufferImp, class EntityType>
        void scatter(MessageBufferImp& buff, const EntityType& element, std::size_t /* n */)
        {
            long long globalIdx = -1;
            buff.read(globalIdx);
            if (globalIdx >= 0)
            {
                globalIndex_.resize();
                globalIndex_[element] = globalIdx;
            }
        }

        //! \brief loop over all internal data handlers and return sum of data
        //! size of given entity
        template<class EntityType>
        std::size_t size(const EntityType& /* en */) const
        { return 1; }

    protected:
        // initialize persistent container from given vector
        void initialize()
        {
            auto idx = cartesianIndex_.begin();
            auto it = gridView_.template begin<0>();
            const auto& endIt = gridView_.template end<0>();

            for (; it != endIt; ++it, ++idx)
                globalIndex_[*it] = *idx;
        }

        // update vector from given persistent container
        void finalize()
        {
            std::vector<long long> newIndex ;
            newIndex.reserve(gridView_.indexSet().size(0));

            auto it = gridView_.template begin<0>();
            const auto& endIt = gridView_.template end<0>();
            for (; it != endIt; ++it)
                newIndex.push_back(globalIndex_[*it].index()) ;

            cartesianIndex_.swap(newIndex);
        }

        GridView gridView_;
        GlobalIndexContainer globalIndex_;
        std::vector<long long>& cartesianIndex_;
    };

public:
    /** \brief dimension of the grid */
    static constexpr long long dimension = Grid::dimension ;

    /** \brief constructor taking grid */
    CartesianIndexMapper(const Grid& grid,
                         const std::array<long long, dimension>& cartDims,
                         const std::vector<long long>& cartesianIndex)
        : grid_(grid)
        , cartesianDimensions_(cartDims)
        , cartesianIndex_(cartesianIndex)
        , cartesianSize_(computeCartesianSize())
    {}

    /** \brief return Cartesian dimensions, i.e. number of cells in each direction  */
    const std::array<long long, dimension>& cartesianDimensions() const
    { return cartesianDimensions_; }

    /** \brief return total number of cells in the logical Cartesian grid */
    long long cartesianSize() const
    { return cartesianSize_; }

    /** \brief return number of cells in the active grid */
    long long compressedSize() const
    { return cartesianIndex_.size(); }

    /** \brief return index of the cells in the logical Cartesian grid */
    long long cartesianIndex(const long long compressedElementIndex) const
    {
        assert(compressedElementIndex < compressedSize());
        return cartesianIndex_[compressedElementIndex];
    }

    /** \brief return index of the cells in the logical Cartesian grid */
    long long cartesianIndex(const std::array<long long, dimension>& coords) const
    {
        long long cartIndex = coords[0];
        long long factor = cartesianDimensions()[0];
        for (long long i=1; i < dimension; ++i) {
            cartIndex += coords[i] * factor;
            factor *= cartesianDimensions()[i];
        }

        return cartIndex;
    }

    /** \brief return Cartesian coordinate, i.e. IJK, for a given cell */
    void cartesianCoordinate(const long long compressedElementIndex, std::array<long long, dimension>& coords) const
    {
        long long gc = cartesianIndex(compressedElementIndex);
        if constexpr (dimension == 3) {
            coords[0] = gc % cartesianDimensions()[0];
            gc /= cartesianDimensions()[0];
            coords[1] = gc % cartesianDimensions()[1];
            coords[2] = gc / cartesianDimensions()[1];
        }
        else if constexpr (dimension == 2) {
            coords[0] = gc % cartesianDimensions()[0];
            coords[1] = gc / cartesianDimensions()[0];
        }
        else if constexpr (dimension == 1)
            coords[0] = gc ;
        else
            throw std::invalid_argument("cartesianCoordinate not implemented for dimension " + std::to_string(dimension));
    }

    template <class GridView>
    std::unique_ptr<GlobalIndexDataHandle<GridView> > dataHandle(const GridView& gridView)
    {
        using DataHandle = GlobalIndexDataHandle<GridView>;
        assert(&grid_ == &gridView.grid());
        return std::make_unique<DataHandle>(gridView, cartesianIndex_);
    }

protected:
    long long computeCartesianSize() const
    {
        long long size = cartesianDimensions()[0];
        for (long long d=1; d<dimension; ++d)
            size *= cartesianDimensions()[d];
        return size ;
    }

    const Grid& grid_;
    const std::array<long long, dimension> cartesianDimensions_;
    std::vector<long long> cartesianIndex_;
    const long long cartesianSize_ ;
};

} // end namespace Dune

#endif // OPM_ALUGRID_CARTESIAN_INDEX_MAPPER_HPP
