#ifndef OPM_CUISTL_CUBLOCKPRECONDITIONER_HEADER
#define OPM_CUISTL_CUBLOCKPRECONDITIONER_HEADER

#include <dune/common/shared_ptr.hh>
#include <memory>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>

namespace Opm::cuistl
{
template <class X, class Y, class C, class P = Dune::PreconditionerWithUpdate<X, Y>>
class CuBlockPreconditioner : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    using domain_type = X;
    using range_type = Y;
    using field_type = typename X::field_type;
    using communication_type = C;


    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param p The sequential preconditioner.
       \param c The communication object for syncing overlap and copy
       data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    CuBlockPreconditioner(const std::shared_ptr<P>& p, const std::shared_ptr<communication_type>& c)
        : m_preconditioner(p)
        , m_communication(c)
    {
    }

    CuBlockPreconditioner(const std::shared_ptr<P>& p, const communication_type& c)
        : m_preconditioner(p)
        , m_communication(Dune::stackobject_to_shared_ptr(c))
    {
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre(X& x, Y& b) override
    {
        m_communication->copyOwnerToAll(x, x); // make dirichlet values consistent
        m_preconditioner->pre(x, b);
    }

    /*!
       \brief Apply the preconditioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply(X& v, const Y& d) override
    {
        m_preconditioner->apply(v, d);
        m_communication->copyOwnerToAll(v, v);
    }


    virtual void update() override
    {
        m_preconditioner->update();
    }

    virtual void post(X& x) override
    {
        m_preconditioner->post(x);
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::overlapping;
    }

private:
    //! \brief a sequential preconditioner
    std::shared_ptr<P> m_preconditioner;

    //! \brief the communication object
    std::shared_ptr<communication_type> m_communication;
};
} // namespace Opm::cuistl
#endif
