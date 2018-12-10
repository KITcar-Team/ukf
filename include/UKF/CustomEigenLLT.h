// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CUSTOM_EIGEN_LLT_H
#define CUSTOM_EIGEN_LLT_H
/**
 * This is taken from Eigen::Cholesky. This had to be moved out because it requires a patch to be used here.
 * Details can be found here: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1477
 */
namespace CustomEigen { 

namespace internal{
template<typename MatrixType, int UpLo> struct LLT_Traits;
}

/** \ingroup Cholesky_Module
  *
  * \class LLT
  *
  * \brief Standard Cholesky decomposition (LL^T) of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the LL^T Cholesky decomposition
  * \param UpLo the triangular part that will be used for the decompositon: Eigen:Lower (default) or Eigen:Upper.
  *             The other triangular part won't be read.
  *
  * This class performs a LL^T Cholesky decomposition of a symmetric, positive definite
  * matrix A such that A = LL^* = U^*U, where L is lower triangular.
  *
  * While the Cholesky decomposition is particularly useful to solve selfadjoint problems like  D^*D x = b,
  * for that purpose, we recommend the Cholesky decomposition without square root which is more stable
  * and even faster. Nevertheless, this standard Cholesky decomposition remains useful in many other
  * situations like generalised eigen problems with hermitian matrices.
  *
  * Remember that Cholesky decompositions are not rank-revealing. This LLT decomposition is only stable on positive definite matrices,
  * use LDLT instead for the semidefinite case. Also, do not use a Cholesky decomposition to determine whether a system of equations
  * has a solution.
  *
  * Example: \include LLT_example.cpp
  * Output: \verbinclude LLT_example.out
  *    
  * \sa Eigen::MatrixBase::llt(), SelfAdjointView::llt(), class LDLT
  */
 /* HEY THIS DOX IS DISABLED BECAUSE THERE's A BUG EITHER HERE OR IN LDLT ABOUT THAT (OR BOTH)
  * Note that during the decomposition, only the upper triangular part of A is considered. Therefore,
  * the strict lower part does not have to store correct values.
  */
template<typename _MatrixType, int _UpLo> class LLT
{
  public:
    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename Eigen::NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3
    typedef typename MatrixType::StorageIndex StorageIndex;

    enum {
      PacketSize = Eigen::internal::packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1,
      UpLo = _UpLo
    };

    typedef internal::LLT_Traits<MatrixType,UpLo> Traits;

    /**
      * \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via LLT::compute(const MatrixType&).
      */
    LLT() : m_matrix(), m_isInitialized(false) {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa LLT()
      */
    explicit LLT(Index size) : m_matrix(size, size),
                    m_isInitialized(false) {}

    template<typename InputType>
    explicit LLT(const Eigen::EigenBase<InputType>& matrix)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_isInitialized(false)
    {
      compute(matrix.derived());
    }

    /** \returns a view of the upper triangular matrix U */
    inline typename Traits::MatrixU matrixU() const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      return Traits::getU(m_matrix);
    }

    /** \returns a view of the lower triangular matrix L */
    inline typename Traits::MatrixL matrixL() const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      return Traits::getL(m_matrix);
    }

    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * Since this LLT class assumes anyway that the matrix A is invertible, the solution
      * theoretically exists and is unique regardless of b.
      *
      * Example: \include LLT_solve.cpp
      * Output: \verbinclude LLT_solve.out
      *
      * \sa solveInPlace(), Eigen::MatrixBase::llt(), SelfAdjointView::llt()
      */
    template<typename Rhs>
    inline const Eigen::Solve<LLT, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      eigen_assert(m_matrix.rows()==b.rows()
                && "LLT::solve(): invalid number of rows of the right hand side matrix b");
      return Eigen::Solve<LLT, Rhs>(*this, b.derived());
    }

    template<typename InputType>
    LLT& compute(const Eigen::EigenBase<InputType>& matrix);

    /** \returns the LLT decomposition matrix
      *
      * TODO: document the storage layout
      */
    inline const MatrixType& matrixLLT() const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      return m_matrix;
    }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    Eigen::ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      return m_info;
    }

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

  protected:
    
    static void check_template_parameters()
    {
      using namespace Eigen;
      EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar);
    }
    
    /** \internal
      * Used to compute and store L
      * The strict upper part is not used and even not initialized.
      */
    MatrixType m_matrix;
    bool m_isInitialized;
    Eigen::ComputationInfo m_info;
};

namespace internal {

template<typename Scalar, int UpLo> struct llt_inplace;

template<typename MatrixType, typename VectorType>
static Eigen::Index llt_rank_update_lower(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma)
{
  using std::sqrt;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::ColXpr ColXpr;
  typedef typename Eigen::internal::remove_all<ColXpr>::type ColXprCleaned;
  typedef typename ColXprCleaned::SegmentReturnType ColXprSegment;
  typedef Eigen::Matrix<Scalar,VectorType::RowsAtCompileTime,1,0,VectorType::MaxRowsAtCompileTime,1> TempVectorType;
  typedef typename TempVectorType::SegmentReturnType TempVecSegment;

  Eigen::Index n = mat.cols();
  eigen_assert(mat.rows()==n && vec.size()==n);

  TempVectorType temp;

  if(sigma>0)
  {
    // This version is based on Givens rotations.
    // It is faster than the other one below, but only works for updates,
    // i.e., for sigma > 0
    temp = sqrt(sigma) * vec;

    for(Eigen::Index i=0; i<n; ++i)
    {
      Eigen::JacobiRotation<Scalar> g;
      g.makeGivens(mat(i,i), -temp(i), &mat(i,i));

      Eigen::Index rs = n-i-1;
      if(rs>0)
      {
        ColXprSegment x(mat.col(i).tail(rs));
        TempVecSegment y(temp.tail(rs));
        apply_rotation_in_the_plane(x, y, g);
      }
    }
  }
  else
  {
    temp = vec;
    RealScalar beta = 1;
    for(Eigen::Index j=0; j<n; ++j)
    {
      RealScalar Ljj = Eigen::numext::real(mat.coeff(j,j));
      RealScalar dj = Eigen::numext::abs2(Ljj);
      Scalar wj = temp.coeff(j);
      RealScalar swj2 = sigma*Eigen::numext::abs2(wj);
      RealScalar gamma = dj*beta + swj2;

      RealScalar x = dj + swj2/beta;
      if (x<=RealScalar(0))
        return j;
      RealScalar nLjj = sqrt(x);
      mat.coeffRef(j,j) = nLjj;
      beta += swj2/dj;

      // Update the terms of L
      Eigen::Index rs = n-j-1;
      if(rs)
      {
        temp.tail(rs) -= (wj/Ljj) * mat.col(j).tail(rs);
        if(gamma != 0)
          mat.col(j).tail(rs) = (nLjj/Ljj) * mat.col(j).tail(rs) + (nLjj * sigma*Eigen::numext::conj(wj)/gamma)*temp.tail(rs);
      }
    }
  }
  return -1;
}

template<typename Scalar> struct llt_inplace<Scalar, Eigen::Lower>
{
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  template<typename MatrixType>
  static Eigen::Index unblocked(MatrixType& mat)
  {
    using std::sqrt;
    
    eigen_assert(mat.rows()==mat.cols());
    const Eigen::Index size = mat.rows();
    for(Eigen::Index k = 0; k < size; ++k)
    {
      Eigen::Index rs = size-k-1; // remaining size

      Eigen::Block<MatrixType,Eigen::Dynamic,1> A21(mat,k+1,k,rs,1);
      Eigen::Block<MatrixType,1,Eigen::Dynamic> A10(mat,k,0,1,k);
      Eigen::Block<MatrixType,Eigen::Dynamic,Eigen::Dynamic> A20(mat,k+1,0,rs,k);

      RealScalar x = Eigen::numext::real(mat.coeff(k,k));
      if (k>0) x -= A10.squaredNorm();
      if (x<=RealScalar(0))
        return k;
      mat.coeffRef(k,k) = x = sqrt(x);
      if (k>0 && rs>0) A21.noalias() -= A20 * A10.adjoint();
      if (rs>0) A21 /= x;
    }
    return -1;
  }

  template<typename MatrixType>
  static Eigen::Index blocked(MatrixType& m)
  {
    eigen_assert(m.rows()==m.cols());
    Eigen::Index size = m.rows();
    if(size<32)
      return unblocked(m);

    Eigen::Index blockSize = size/8;
    blockSize = (blockSize/16)*16;
    blockSize = (std::min)((std::max)(blockSize,Eigen::Index(8)), Eigen::Index(128));

    for (Eigen::Index k=0; k<size; k+=blockSize)
    {
      // partition the matrix:
      //       A00 |  -  |  -
      // lu  = A10 | A11 |  -
      //       A20 | A21 | A22
      Eigen::Index bs = (std::min)(blockSize, size-k);
      Eigen::Index rs = size - k - bs;
      Eigen::Block<MatrixType,Eigen::Dynamic,Eigen::Dynamic> A11(m,k,   k,   bs,bs);
      Eigen::Block<MatrixType,Eigen::Dynamic,Eigen::Dynamic> A21(m,k+bs,k,   rs,bs);
      Eigen::Block<MatrixType,Eigen::Dynamic,Eigen::Dynamic> A22(m,k+bs,k+bs,rs,rs);

      Eigen::Index ret;
      if((ret=unblocked(A11))>=0) return k+ret;
      if(rs>0) A11.adjoint().template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(A21);
      if(rs>0) A22.template selfadjointView<Eigen::Lower>().rankUpdate(A21,-1); // bottleneck
    }
    return -1;
  }

  template<typename MatrixType, typename VectorType>
  static Eigen::Index rankUpdate(MatrixType& mat, const VectorType& vec, const RealScalar& sigma)
  {
    return CustomEigen::internal::llt_rank_update_lower(mat, vec, sigma);
  }
};
  
template<typename Scalar> struct llt_inplace<Scalar, Eigen::Upper>
{
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;

  template<typename MatrixType>
  static EIGEN_STRONG_INLINE Eigen::Index unblocked(MatrixType& mat)
  {
    Eigen::Transpose<MatrixType> matt(mat);
    return llt_inplace<Scalar, Eigen::Lower>::unblocked(matt);
  }
  template<typename MatrixType>
  static EIGEN_STRONG_INLINE Eigen::Index blocked(MatrixType& mat)
  {
    Eigen::Transpose<MatrixType> matt(mat);
    return llt_inplace<Scalar, Eigen::Lower>::blocked(matt);
  }
  template<typename MatrixType, typename VectorType>
  static Eigen::Index rankUpdate(MatrixType& mat, const VectorType& vec, const RealScalar& sigma)
  {
    Eigen::Transpose<MatrixType> matt(mat);
    return llt_inplace<Scalar, Eigen::Lower>::rankUpdate(matt, vec.conjugate(), sigma);
  }
};

template<typename MatrixType> struct LLT_Traits<MatrixType,Eigen::Lower>
{
  typedef const Eigen::TriangularView<const MatrixType, Eigen::Lower> MatrixL;
  typedef const Eigen::TriangularView<const typename MatrixType::AdjointReturnType, Eigen::Upper> MatrixU;
  static inline MatrixL getL(const MatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const MatrixType& m) { return MatrixU(m.adjoint()); }
  static bool inplace_decomposition(MatrixType& m)
  { return llt_inplace<typename MatrixType::Scalar, Eigen::Lower>::blocked(m)==-1; }
};

template<typename MatrixType> struct LLT_Traits<MatrixType,Eigen::Upper>
{
  typedef const Eigen::TriangularView<const typename MatrixType::AdjointReturnType, Eigen::Lower> MatrixL;
  typedef const Eigen::TriangularView<const MatrixType, Eigen::Upper> MatrixU;
  static inline MatrixL getL(const MatrixType& m) { return MatrixL(m.adjoint()); }
  static inline MatrixU getU(const MatrixType& m) { return MatrixU(m); }
  static bool inplace_decomposition(MatrixType& m)
  { return llt_inplace<typename MatrixType::Scalar, Eigen::Upper>::blocked(m)==-1; }
};

} // end namespace internal

/** Computes / recomputes the Cholesky decomposition A = LL^* = U^*U of \a matrix
  *
  * \returns a reference to *this
  *
  * Example: \include TutorialLinAlgComputeTwice.cpp
  * Output: \verbinclude TutorialLinAlgComputeTwice.out
  */
template<typename MatrixType, int _UpLo>
template<typename InputType>
LLT<MatrixType,_UpLo>& LLT<MatrixType,_UpLo>::compute(const Eigen::EigenBase<InputType>& a)
{
  check_template_parameters();
  
  eigen_assert(a.rows()==a.cols());
  const Eigen::Index size = a.rows();
  m_matrix.resize(size, size);
  m_matrix = a.derived();

  m_isInitialized = true;
  bool ok = Traits::inplace_decomposition(m_matrix);
  m_info = ok ? Eigen::Success : Eigen::NumericalIssue;

  return *this;
}

template<typename Derived>
inline const LLT<typename Eigen::MatrixBase<Derived>::PlainObject, Eigen::Upper>
llt(const Eigen::MatrixBase<Derived>& m)
{
  return LLT<typename Eigen::MatrixBase<Derived>::PlainObject, Eigen::Upper>(m.derived());
}

} // end namespace CustomEigen

#endif // CUSTOM_EIGEN_LLT_H
