import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
# Assuming timer_wrapper as tw is correctly implemented with a timeit decorator
import timer_wraper as tw

input_dir = 'Images'
output_dir = 'assignment1_output'

def ensure_dir(directory):
    """Ensure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_to_binary(img):
    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Calculate the cumulative sum of the histogram
    cumulative_hist = np.cumsum(hist)
    # Find the midpoint of the histogram
    midpoint = np.argmax(cumulative_hist >= np.sum(hist) / 2)
    # Threshold the image using the midpoint
    _, binary_img = cv2.threshold(img, midpoint, 255, cv2.THRESH_BINARY)
    return binary_img

def black_and_white(input_file, output_file):
    ensure_dir(os.path.dirname(output_file))
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    binary_img = convert_to_binary(img)
    cv2.imwrite(output_file, binary_img)

def blur(input_file, output_file):
    ensure_dir(os.path.dirname(output_file))
    img = cv2.imread(input_file)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(output_file, blurred)

def noise(input_file, output_file):
    ensure_dir(os.path.dirname(output_file))
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    binary_img = convert_to_binary(img)
    black_pixel_count = np.sum(binary_img == 0)
    noise_pixel_count = int(0.1 * black_pixel_count)
    noise_img = add_noise(binary_img, noise_pixel_count)
    cv2.imwrite(output_file, noise_img)

def add_noise(binary_img, noise_pixel_count):
    noise_img = binary_img.copy()
    mask = np.zeros_like(binary_img)
    idx = (np.random.choice(range(mask.shape[0]), size=noise_pixel_count, replace=True), 
           np.random.choice(range(mask.shape[1]), size=noise_pixel_count, replace=True))
    noise_img[idx] = 255 - noise_img[idx]  # Invert the color to add "noise"
    return noise_img

@tw.timeit
def parallel(func, cpu_count):
    pool = Pool(cpu_count)
    data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir) if file.endswith(('.jpg', '.png'))]
    for _, output_file in data:
        ensure_dir(os.path.dirname(output_file))
    pool.starmap(func, data)
    pool.close()
    pool.join()

@tw.timeit
def multithreaded(func, thread_count):
    print(f'Executing {func.__name__}')
    pool = ThreadPool(thread_count)
    data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir) if file.endswith(('.jpg', '.png'))]
    for _, output_file in data:
        ensure_dir(os.path.dirname(output_file))
    pool.starmap(func, data)
    pool.close()
    pool.join()

def execute(executor, counts, name):
    for count in counts:
        print(f'Count: {count}')
        executor(black_and_white, count)
        executor(blur, count)
        executor(noise, count)

if __name__ == '__main__':
    # Ensure base output directory exists
    ensure_dir(output_dir)

    # Execute parallel and multithreaded tasks
    execute(parallel, range(2, cpu_count() + 1), 'CPUs')
    execute(multithreaded, range(2, 400, 20), 'threads')
    
