#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>


// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

struct rand_functor {
  const int a;
  rand_functor(int _a) : a(_a) {} 
  __host__ __device__ 
  int  operator()() const { 
  return rand() % a; 
 }}; 

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  //n value
  int n = histogram.size();

  // find largest bin size 
  int largest_bin = 0;
  for (int i =0; i< n; i++){
    if (largest_bin < histogram[i]){
      largest_bin = histogram[i];
    }
  }


  printf("Max bin capacity for  N = %d is %d\n", n, largest_bin); 
  
 
 }

int main(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  for (int n =2; n< (2<<20); n *=2){ 
	thrust::host_vector<int> input(n); 
	thrust::generate(input.begin(), input.end(), rand_functor(n)); 
	thrust::device_vector<int> histogram;
	dense_histogram(input, histogram);
	}	 
  
  return 0;

  }
  
  
