// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <queue>
#include <random>
#include <opencv2/core/core.hpp>
//#include "uniform_random_generator.h"
#include "sampler.h"
#include <time.h>

namespace gcransac
{
	namespace sampler
	{
		class GumbelSoftmaxSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			//size_t sample_size;
			size_t point_number;
			double tau;
			std::vector<double>logits;
			std::random_device rd{};
			std::mt19937 gen{rd()};
			std::extreme_value_distribution<> gumbel_dist{0.0, 1.0};	
			std::vector<size_t> idx;
			std::priority_queue<std::pair<double, size_t>, 
				std::vector<std::pair<double, size_t>> > processing_queue;

			//std::priority_queue<<double> processing_queue;

		public:
			explicit GumbelSoftmaxSampler(const cv::Mat * const container_,
				const std::vector<double> &inlier_probabilities_,
				const size_t sample_size_,
				const double tau_ = 1.0)
				//sample_size(sample_size_))
				//point_number(container_->rows),
				: Sampler(container_),
					logits(inlier_probabilities_),
					tau(tau_)
			{
				//std::cout<<"debug1"<<sample_size_<<std::endl;
				//std::cout<<"debug2"<<container_->rows<<std::endl;
				//std::cout<<"logits"<<inlier_probabilities_[0]<<"_"<<logits.size()<<std::endl;
				clock_t start, finish;
				if (inlier_probabilities_.size() != container_->rows)
				{
					fprintf(stderr, "The number of correspondences (%d) and the number of provided probabilities (%d) do not match.",
						container_->rows, 
						inlier_probabilities_.size());
					// emplace ones if there is no logits provided
					//std::vector<double> gumbels;
					for (size_t i = 0; i < logits.size(); ++i)
					{
						logits.emplace_back(1.0);
						//std::cout<<"no logitttts input"<<std::endl;

					}

					//return;
				}
				//idx.reserve(logits.size());
				start = clock();
				for (size_t i = 0; i < logits.size(); ++i)
				{	
				
					//idx.emplace_back(i);
					//gumbels.emplace_back(logits[i]);
					//std::cout<<idx[i]<<std::endl;
					double gumbel = (logits[i] + gumbel_dist(gen)) * (1/tau);
					processing_queue.emplace(std::make_pair(gumbel, i));

				}
				finish = clock();
				//std::cout<<processing_queue.size()<<std::endl;
				//std::cout<<"time cost GumbelSoftmaxSampler:"<<double(finish-start)<<std::endl;
			//
				// Initializing the base class
				initialized = initialize(container_);
			}

			~GumbelSoftmaxSampler() {}

			const std::string getName() const { return "Gumbel Softmax Sampler"; }

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				

				return true;
			}

			void reset()
			{
				//gumbels.clear();
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);

			OLGA_INLINE void update(
				const size_t* const subset_,
				const size_t &sample_size_,
				const size_t& iteration_number_,
				const double &inlier_ratio_)
			{
				//std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>> > processing_queue_tmp;
				//processing_queue=processing_queue_tmp; //
				//processing_queue.swap(processing_queue_tmp);
				//std::cout<<"before update"<<processing_queue.size()<<std::endl;

				size_t idx;
				double updated_inlier_ratio;
				for (size_t i = 0; i < sample_size_; ++i)
				{
					idx = subset_[i];
					updated_inlier_ratio = (logits[idx] + gumbel_dist(gen)) * (1 / tau);
					processing_queue.emplace(std::make_pair(updated_inlier_ratio, idx));
				}

				/*for (size_t i = 0; i < logits.size(); ++i)
				{
					// std::cout<<"the gumbel generator:""<<gumbel_dist(gen)<<std::endl;
					double updated_inlier_ratio = (logits[i] + gumbel_dist(gen)) * (1/tau);
					//double random = 
					//std::cout<<"d "<<(randomness_rand_max * static_cast<double>(rand()) - randomness_2)<<std::endl;
					//std::cout<<"d "<<pres[i]<<std::endl;
					//processing_queue.pop();
				}
				finish=clock();*/
				//std::cout<<"time cost update:"<<double(finish-start)<<std::endl;
				
			}
		};

		

		OLGA_INLINE bool GumbelSoftmaxSampler::sample(
			const std::vector<size_t>& pool_,
			size_t* const subset_,
			size_t sample_size_)
		{

			//if (sample_size_ != sample_size)
			//	{
			//		fprintf(stderr, "An error occured when sampling.\n");
			//		return false;
			//	}
			// calculate gumbels from logits and Gumbel distribution, saving for selecting the best K points


			//std::vector<double> gumbels;
			//gumbels.reserve(logits.size());
			//for (size_t i = 0; i < logits.size(); ++i)
			//{
				//std::cout<<"debug3"<<logits[i]<<std::endl;
			//	double gumbel = (logits[i] + gumbel_dist(gen)) * (1/tau);
				//double gumbel = logits[i] + gumbel_dist(gen) * (1/tau);
			//	gumbels[i] = gumbel;
				//gumbels.emplace_back(gumbel);
				//idx.emplace_back(i);
				//processing_queue.emplace(std::make_pair(gumbel, i));

			//}
			// y_soft calculation, won't change anythig in testing case
				//double MAX= gumbels[0];
				//for (auto x:gumbels)
				//{
				//	MAX = max(x, MAX);
				//}

				//double sum = 0;
				//for (auto x:gumbels)
				//{
				//	sum += exp(x-MAX);

				//}
				//std::vector<double> y_soft;
				//for  (auto x:gumbels)
				//{
				//	y_soft.push_back(exp(x-MAX)/sum);

				//}
			// find the best k points
			
			//initialize the indices
			//std::vector<size_t> idx(gumbels.size());
			//for (int i=0; i !=idx.size(); ++i) idx[i] = i;
			
			// reordering
			//sort(idx.begin(), idx.end(), 
			//	[&gumbels](size_t i1, size_t i2){return gumbels[i1] > gumbels[i2]; });
			//utils::sort_indices(gumbels, idx);
			//std::cout<<"sorted"<<idx.size()<<std::endl;
			
			//std::cout<<"-"<< idx.size()<<std::endl;
			//clock_t start, finish;
			//start = clock();

			for (size_t i = 0; i < sample_size_; ++i)
			{
				const auto& item = processing_queue.top();
				subset_[i] = item.second;
				//std::cout<<item.second<<std::endl;
				//subset_[i] = pool_[item];//idx[i]
				processing_queue.pop();
			}
			//finish = clock();
			//std::cout<<"time cost sample:"<<double(finish-start)<<std::endl;
			return true;
			
		}
	}
}
