// Compile with cmake (CMakeLists.txt is provided) or with the following lines in bash:
// g++ -c -fPIC libautosim.cpp -o libautosim.o
// g++ -shared -Wl,-soname,libautosim.so -o libautosim.so libautosim.o


#include <string>
#include <sstream>
#include <iostream>
#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <float.h>
#include <limits>
#include <cmath>
#include <limits>


#include <bitset>
#include <boost/math/distributions/binomial.hpp>
#include "cpp_libs/libOrsa/orsa_fundamental.hpp"
#include "cpp_libs/libOrsa/orsa_homography.hpp"

#define FullDescDim 6272
#define VecDescDim 128 // as in (x,x,128) output of the network
#define SameKPThres 4


struct TargetNode
{
  const int TargetIdx; float sim_with_query;
  TargetNode(int Idx, float sim):TargetIdx(Idx){ this->sim_with_query = sim; };
  bool operator >(const TargetNode & kp) {return ( this->sim_with_query>kp.sim_with_query );};
  bool operator ==(const TargetNode & kp) {return ( this->sim_with_query==kp.sim_with_query );};
};

struct QueryNode
{
  const int QueryIdx;
  std::list<QueryNode>::iterator thisQueryNodeOnList; //pointer on list
  float first_sim, last_sim;
  std::list<TargetNode> MostSimilar_TargetNodes;

  void Add_TargetNode(int it, float sim, int MaxTnodes_num)
  {
    TargetNode tn(it,sim);
    std::list<TargetNode>::iterator target_iter;
    for(target_iter = MostSimilar_TargetNodes.begin(); target_iter != MostSimilar_TargetNodes.end(); ++target_iter)
      if ( tn > *target_iter )
        break;

    MostSimilar_TargetNodes.insert( target_iter, tn );
    if (MaxTnodes_num>0 && MostSimilar_TargetNodes.size()>MaxTnodes_num)
      MostSimilar_TargetNodes.pop_back();
      last_sim = (--MostSimilar_TargetNodes.end())->sim_with_query;
    first_sim = MostSimilar_TargetNodes.begin()->sim_with_query;  
  };

  QueryNode(int Idx):QueryIdx(Idx){ first_sim = -1; last_sim = -1; };
};

struct DescStatsClass
{
  float norm, max, min, mean, sigma;
  std::bitset<FullDescDim> AID;
  std::bitset<VecDescDim> AIDbyVec [FullDescDim/VecDescDim];

  DescStatsClass()
  {
    norm = 0.0f, mean = 0.0f, sigma = 0.0f;
    max = -std::numeric_limits<float>::infinity();
    min = std::numeric_limits<float>::infinity();
  };
};


int CountItFast(const DescStatsClass & q, const DescStatsClass & t, int thres)
{
  int xor_opp = 0;
  for (int v = 0; (v < FullDescDim/VecDescDim && (xor_opp < thres)); v++)
    xor_opp +=  (q.AIDbyVec[v] ^ t.AIDbyVec[v]).count();
  return(xor_opp);
}


float FastSimi(int iq, float* DescsQuery, DescStatsClass* QueryStats, int it, float* DescsTarget, DescStatsClass* TargetStats, float simi_thres)
{
  float m_sq = QueryStats[iq].max * TargetStats[it].max;
  float norms_prod = QueryStats[iq].norm * TargetStats[it].norm;
  float dynamic_thres = (simi_thres * norms_prod) - FullDescDim * m_sq;
  float SP = 0.0;
  int qpos = iq*FullDescDim, tpos = it*FullDescDim;
  for (int i = 0; (i < FullDescDim); i++)
  {
    SP +=  DescsQuery[qpos + i] * DescsTarget[tpos + i];
  }
  return(SP/norms_prod);
};

DescStatsClass LoadInfo(int iq, float* DescsQuery)
{
  DescStatsClass ds;
  float val;
  int qpos = iq*FullDescDim;
  for (int i = 0; (i < FullDescDim); i++)
  {
    val = DescsQuery[qpos + i];
    ds.norm += val * val;
  }
  
  for (int i = 0; (i < FullDescDim); i++)
    if (DescsQuery[qpos + i]>=0)
    {
      ds.AID.set(i);
    }

  for (int v = 0; (v < FullDescDim/VecDescDim); v++)
  {
    int vpos = VecDescDim*v;
    for (int i = 0; (i < VecDescDim); i++)
      {
        if (DescsQuery[qpos + vpos + i]>=0)
          ds.AIDbyVec[v].set(i);
      }
  }

  ds.norm = std::sqrt(ds.norm);
  return(ds);
};

// this gives P(X>=k) when X is a binomial distribution of parameters N and p
float* BinomialSurvivals(int N, float p)
{
  float * survivals = new float[N+1];
  // binomial distribution object:
  boost::math::binomial bino(N,p);
  // survival values
  for (int x=0; x<=N; x++)
    survivals[x] = cdf(complement(bino, x));
  return(survivals);
}


struct KeypointClass{
  float x,y;
};


struct MatchersClass
{
  std::list<QueryNode> QueryNodes;
  const int k; // k as in knn, if <=0 then we store all matches having a sim above sim_thres
  const float sim_thres;
  const int Nvec;
  std::vector<KeypointClass> QueryKPs, TargetKPs;
  
  int* FilteredIdxMatches;
  int N_FilteredMatches;


  int BinDesc_GetAlarmThreshold(int Nt)
  {
    float *survivals = BinomialSurvivals(VecDescDim, 0.5);
    int alarmThres = 0.0;
    for (int i = 0; i <= VecDescDim; i++)
      if (Nt * survivals[i] <= 1)
      {
        alarmThres = (int)(i * Nvec);
        break;
      }
    return (alarmThres);
  }

  MatchersClass(int knn_num, float sim_thres) : k(knn_num), sim_thres(sim_thres), Nvec((int)(FullDescDim / VecDescDim)){};

  void KnnMatcher(float *DescsQuery, int Nquery, float *DescsTarget, int Ntarget, int FastCode)
  {
    DescStatsClass *QueryStats = new DescStatsClass[Nquery], *TargetStats = new DescStatsClass[Ntarget];
#pragma omp parallel for default(shared)
    for (int iq = 0; iq < Nquery; iq++)
      QueryStats[iq] = LoadInfo(iq, DescsQuery);
#pragma omp parallel for default(shared)
    for (int it = 0; it < Ntarget; it++)
      TargetStats[it] = LoadInfo(it, DescsTarget);

    switch (FastCode)
    {
      case 0:
      { // BigAID
        // std::cout<<"---> Brute Force Angle Comparisons"<<std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          for (int it = 0; it < Ntarget; it++)
          {
            float updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = FastSimi(iq, DescsQuery, QueryStats, it, DescsTarget, TargetStats, updated_sim_thres);
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
  #pragma omp critical
          {
            QueryNodes.push_back(qn);
            std::list<QueryNode>::iterator itqn = --QueryNodes.end();
            itqn->thisQueryNodeOnList = itqn;
          }
        }
        break;
      } // end of BigAID
      case 1:
      { // model new AID
        // std::cout << "---> Full sign comparisons with bitset!!!" << std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          float updated_sim_thres;
          for (int it = 0; it < Ntarget; it++)
          {
            // This is like counting bits after an XNOR opperation on both binary descriptors
            updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = (float) ( FullDescDim - CountItFast(QueryStats[iq], TargetStats[it], FullDescDim - updated_sim_thres) );
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
          {
  #pragma omp critical
            {
              QueryNodes.push_back(qn);
              std::list<QueryNode>::iterator itqn = --QueryNodes.end();
              itqn->thisQueryNodeOnList = itqn;
            }
          }
        }
        break;
      } // end of AID
      case 2:
      { // model AID
        // std::cout << "---> Full sign comparisons with bitset!!!" << std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          for (int it = 0; it < Ntarget; it++)
          {
            // This is like counting bits after an XNOR opperation on both binary descriptors
            int concor = FullDescDim - (QueryStats[iq].AID ^ TargetStats[it].AID).count();
            float updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = (float)concor;
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
          {
  #pragma omp critical
            {
              QueryNodes.push_back(qn);
              std::list<QueryNode>::iterator itqn = --QueryNodes.end();
              itqn->thisQueryNodeOnList = itqn;
            }
          }
        }
        break;
      } // end of AID
    } // end of the switch
  }

private:
  MatchersClass() : k(0), sim_thres(0.0), Nvec(0){};
};

float max_euclidean_dist(const Match & m1, const Match & m2)
{
  float left_dist = std::sqrt( std::pow(m1.x1-m2.x1,2.0) + std::pow(m1.y1-m2.y1,2.0) );
  float right_dist = std::sqrt( std::pow(m1.x2-m2.x2,2.0) + std::pow(m1.y2-m2.y2,2.0) );
  if (left_dist>right_dist)
    return left_dist;
  else
    return right_dist;
}

std::vector<Match> UniqueFilter(const std::vector<Match>& matches)
{
  std::vector<Match> uniqueM;
  bool *duplicatedM = new bool[matches.size()];
  float best_sim;
  int bestidx;
  for (int i =0; i<matches.size(); i++)
    duplicatedM[i] = false;
  for (int i =0; i<matches.size(); i++)
  {
    if (duplicatedM[i])
      continue;
    best_sim = matches[i].similarity;
    bestidx = i;
    for(int j=i+1; j<matches.size();j++)
    {
      // std::cout<< max_euclidean_dist(matches[i],matches[j])<<std::endl;
      if ( !duplicatedM[j] && max_euclidean_dist(matches[i],matches[j])<SameKPThres )
      {
        duplicatedM[j] = true;
        if (best_sim<matches[j].similarity)
        {
          bestidx = j;
          best_sim = matches[j].similarity;
        }
      }
    }
    uniqueM.push_back(matches[bestidx]);
  }
  return uniqueM;
}

void ORSA_Filter(std::vector<Match>& matches, bool* MatchMask, float* T, int w1,int h1,int w2,int h2, bool Fundamental, const double & precision, bool verb)
{
  libNumerics::matrix<double> H(3,3);
  std::vector<int> vec_inliers;
  double nfa;
  const float nfa_max = -2;
  const int ITER_ORSA=10000;
  if (Fundamental)
    orsa::orsa_fundamental(matches, w1,h1,w2,h2, precision, ITER_ORSA,H, vec_inliers,nfa,verb);
  else
     orsa::ORSA_homography(matches, w1,h1,w2,h2, precision, ITER_ORSA,H, vec_inliers,nfa,verb);

  for (int cc = 0; cc < matches.size(); cc++ )
    MatchMask[cc] = false;

  if ( nfa < nfa_max )
  {
    for (int vi = 0; vi < vec_inliers.size(); vi++ )
      MatchMask[vec_inliers[vi]] = true;
    H /= H(2,2);

    int t = 0;
    for(int i = 0; i < H.nrow(); ++i)
        for (int j = 0; j < H.ncol(); ++j)
            T[t++] = H(i,j);
    if (verb)
    {
        printf("The two images match! %d matchings are identified. log(nfa)=%.2f.\n", (int) vec_inliers.size(), nfa);
        if (Fundamental)
          std::cout << "*************** Fundamental **************"<< std::endl;
        else
          std::cout << "*************** Homography ***************"<< std::endl;
        std::cout << H <<std::endl;
        std::cout << "******************************************"<< std::endl;
    }
  }
  else
  {
    if (verb)
        printf("The two images do not match. The matching is not significant:  log(nfa)=%.2f.\n", nfa);
  }
}


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{


  int NumberOfFilteredMatches(MatchersClass* M)
  {
    return M->N_FilteredMatches;
  }

  void ArrayOfFilteredMatches(MatchersClass* M, int* arr)
  {
    for (int i=0; i<3*M->N_FilteredMatches;i++)
      arr[i] = M->FilteredIdxMatches[i];
  }

  void GeometricFilterFromNodes(MatchersClass* M, float* T, int w1,int h1,int w2,int h2, int type, float precision, bool verb)
  {
    std::vector<Match> matches;    
    for(std::list<QueryNode>::const_iterator iq = M->QueryNodes.begin(); iq != M->QueryNodes.end(); ++iq) 
      for(std::list<TargetNode>::const_iterator it = iq->MostSimilar_TargetNodes.begin(); it != iq->MostSimilar_TargetNodes.end(); ++it) 
    {
      Match match1;      
      match1.x1 = M->QueryKPs[iq->QueryIdx].x;
      match1.y1 = M->QueryKPs[iq->QueryIdx].y;
      match1.x2 = M->TargetKPs[it->TargetIdx].x;
      match1.y2 = M->TargetKPs[it->TargetIdx].y;
      match1.similarity = it->sim_with_query;
      match1.Qidx = iq->QueryIdx;
      match1.Tidx = it->TargetIdx;
      matches.push_back(match1);
    }
    
    matches = UniqueFilter(matches);
    bool* MatchMask = new bool[matches.size()];
    switch (type)
    {
      case 0: //Homography
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, false, (double)precision, verb);
        break;
      }
      case 1: // Fundamental
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, true, (double)precision, verb);
        break;
      }
    }

    M->N_FilteredMatches = 0;
    for (int cc = 0; cc < matches.size(); cc++ )
      if (MatchMask[cc])
        M->N_FilteredMatches++;
    M->FilteredIdxMatches = new int[3*M->N_FilteredMatches];
    int fcc = 0;
    for (int cc = 0; cc < matches.size(); cc++ )
      if (MatchMask[cc])
        {
          M->FilteredIdxMatches[3*fcc] = matches[cc].Qidx;
          M->FilteredIdxMatches[3*fcc+1] = matches[cc].Tidx;
          M->FilteredIdxMatches[3*fcc+2] = matches[cc].similarity;
          fcc++;
        }
  }

  void GeometricFilter(float* scr_pts, float* dts_pts, bool* MatchMask, float* T, int N, int w1,int h1,int w2,int h2, int type, float precision, bool verb)
  {
    std::vector<Match> matches;
    for (int cc = 0; cc < N; cc++ )
    {
        Match match1;
        match1.x1 = scr_pts[cc*2];
        match1.y1 = scr_pts[cc*2+1];
        match1.x2 = dts_pts[cc*2];
        match1.y2 = dts_pts[cc*2+1];

        matches.push_back(match1);
    }

    switch (type)
    {
      case 0: //Homography
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, false, (double)precision, verb);
        break;
      }
      case 1: // Fundamental
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, true, (double)precision, verb);
        break;
      }
    }
  }

  MatchersClass* newMatcher(int k, int full_desc_dim, float sim_thres)
  {
    if (full_desc_dim!=FullDescDim)
      std::cout<<"Desc dims don't match ("<<full_desc_dim<<"!="<<FullDescDim<<")"<<std::endl;
    return ( new MatchersClass(k, (float) sim_thres) );
  }

  void KnnMatcher(MatchersClass* M, float* Query_pts, float* DescsQuery, int Nquery, float* Target_pts, float* DescsTarget, int Ntarget, int FastCode)
  {
    M->QueryNodes.clear();
    M->QueryKPs.clear();
    M->TargetKPs.clear();
    KeypointClass kp;
    for (int cc = 0; cc < Nquery; cc++ ){ 
      kp.x = Query_pts[cc*2]; kp.y = Query_pts[cc*2+1];
      M->QueryKPs.push_back( kp );
    }
    for (int cc = 0; cc < Ntarget; cc++ ){ 
      kp.x = Target_pts[cc*2]; kp.y = Target_pts[cc*2+1];
      M->TargetKPs.push_back( kp );
    }
    
    M->KnnMatcher(DescsQuery, Nquery, DescsTarget, Ntarget, FastCode);
  }

  int GetQueryNodeLength(QueryNode* qn)
  {
    return(qn->MostSimilar_TargetNodes.size());
  }

  QueryNode* LastQueryNode(MatchersClass* M)
  {
    if (M->QueryNodes.begin()!=M->QueryNodes.end())
      return(&*(--M->QueryNodes.end()));
    else
      return(0);
  }

  QueryNode* FirstQueryNode(MatchersClass* M)
  {
    if (M->QueryNodes.begin()!=M->QueryNodes.end())
      return(&*M->QueryNodes.begin());
    else
      return(0);
  }

  QueryNode* NextQueryNode(MatchersClass* M, QueryNode* qn)
  {
    if (qn!=0 && ++qn->thisQueryNodeOnList!=M->QueryNodes.end())
      return(&*(++qn->thisQueryNodeOnList));
    else
      return(0);
  }

  QueryNode* PrevQueryNode(MatchersClass* M, QueryNode* qn)
  {
    if (qn!=0 && qn->thisQueryNodeOnList!=M->QueryNodes.begin())
      return(&*(--qn->thisQueryNodeOnList));
    else
      return(0);
  }

  void GetData_from_QueryNode(QueryNode* qn, int* QueryIdx, int *TargetIdxes, float* simis)
  {
    QueryIdx[0] = qn->QueryIdx;
    int i = 0;
    for(std::list<TargetNode>::const_iterator it = qn->MostSimilar_TargetNodes.begin(); it != qn->MostSimilar_TargetNodes.end(); ++it)
    {
      TargetIdxes[i] = it->TargetIdx;
      simis[i] = it->sim_with_query;
      i++;
    }
  }




  void FastMatCombi(int N, float* bP, int* i1_list, int *i2_list, float *patches1, float *patches2, int MemStepImg, int* last_i1_list, int *last_i2_list)
  {
    int MemStepBlock = 2*MemStepImg;
    #pragma omp parallel for firstprivate(MemStepImg, MemStepBlock, N)
    for (int k = 0; k<N; k++)
    {
      int i1 = i1_list[k];
      int i2 = i2_list[k];

      if (last_i1_list[k]!=i1)
        for (int i = 0; i<MemStepImg;i++)
          bP[k*MemStepBlock + 2*i] = patches1[i1*MemStepImg + i];

      if (last_i2_list[k]!=i2)
        for (int i = 0; i<MemStepImg;i++)
          bP[k*MemStepBlock + 2*i+1] = patches2[i2*MemStepImg + i];
    }
  }
}
