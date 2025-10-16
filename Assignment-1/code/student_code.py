import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  length = filter.shape[0]
  width = filter.shape[1]

  image_length = image.shape[0]
  image_width = image.shape[1]
  channels = image.shape[2]

  l_pad=(length-1)//2
  w_pad=(width-1)//2

  padded_image=np.pad(image,((l_pad,l_pad),(w_pad,w_pad),(0,0)),mode='constant',constant_values=0)

  filtered_image = np.zeros_like(image,dtype=np.float32)

  for c in range(channels):
    for i in range(image_length):
      for j in range(image_width):
        window=padded_image[i:i+length,j:j+width,c]
        filtered_image[i,j,c]=np.sum(np.multiply(window,filter))

  filtered_image=np.clip(filtered_image,0,255)
  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  hybrid_image = low_frequencies + high_frequencies

  return low_frequencies, high_frequencies, hybrid_image
