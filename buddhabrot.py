import numpy
import pdb
import Image
import datetime
import joblib
import argh
import os

# in-house img lib
# import gtkutils.img_util as iu

buddha_cache = {}

# one iteration of underlying function
# cache could blow up memory.. be careful
# cache probably only useful for large iteration paths
def iterate_cached(z, c):
    key = (z,c)
    if key in buddha_cache:
        return buddha_cache[key]
    val = z**2 + c
    buddha_cache[key] = val
    return val

def iterate(z, c):
    return z**2 + c

def draw_path(array, path, resolution):
    path = numpy.asarray(path)
    path *= resolution           # scale to pixels
    path += array.shape[0] / 2   # additive offset for centering

    # should replace this for-loop
    for xi, yi in path:
        array[xi, yi] += 1
    return array

def draw_global_path(path, resolution):
    global hits
    path = numpy.asarray(path)
    path *= resolution           # scale to pixels
    path += hits.shape[0] / 2   # additive offset for centering

    # should replace this for-loop
    for xi, yi in path:
        hits[xi, yi] += 1

# computes path of point with mandlebrot function
def compute_path(x, y, max_x_units, max_y_units, max_its):
    x_start = x
    y_start = y

    c = complex(x, y)
    z = 0

    path = []

    escaped = False
    for it in xrange(int(max_its)):
        z = z**2 + c

        if z.real >= max_x_units or z.real < -max_x_units \
           or z.imag >= max_y_units or z.imag < -max_y_units:
            return path

        path.append((z.real, z.imag))

    return []

# avoids keeping all the paths around in memory by writing them 
# to the bins after computation 
def compute_and_draw_path(resolution, *args):
    global hits
    path = compute_path(*args)
    if len(path) > 0:
        draw_global_path(path, resolution)

# global array to store the bin counts in
hits = numpy.zeros((0,0))


# additive computation. can compute 
# from previous files
def multi_bb(max_its = 100,
             all_hits = None):

    n_prev_its = 0
    if all_hits is not None:
        assert(os.path.isfile(all_hits))

        bn = os.path.basename(all_hits)
        # only run from prev multis
        assert(bn.split('_')[0] == 'multi')

        n_prev_its = int(bn.split('_')[2])
        all_hits = numpy.load(all_hits)['arr_0']

    for i in range(max_its):
        print "on iteration {}/{}".format(i, max_its - 1)
        last_fn, last_hits = main(save = False)

        # grab the shape and dtype info to init
        if all_hits is None:
            all_hits = numpy.zeros_like(last_hits)
        all_hits += last_hits


    ts = str(datetime.datetime.now()).split()[1][:10]
    fn_base = 'multi_buddhabrot_{}_iterations_{}'.format(max_its + n_prev_its, ts)

    numpy_fn = fn_base + '.npz'
    numpy.savez_compressed(numpy_fn, all_hits)

    # iu.v(((all_hits / float(all_hits.max()))).astype('float64'))

    fn = save_hits(all_hits, fn_base)
    return fn

def save_hits(hit_arr, fn_base):        
    final_fn = fn_base + '.png'
    Image.fromarray(((hit_arr / float(hit_arr.max()))* 255).astype('uint8')).save(final_fn)
    return final_fn

# very naive way to compute a buddhabrot
def main(save = True,
         width = 1600,
         height = 1600,
         pixels_per_unit = 400.0,
         n_points = 10000,
         max_its_per_point = 100):

    # nonsquare arrays untested
    
    global hits
    hits = numpy.zeros((width, height), dtype = numpy.float64)


    max_x_pixels = width / 2
    max_y_pixels = height / 2

    max_x_units = max_x_pixels / pixels_per_unit
    max_y_units = max_y_pixels / pixels_per_unit

    x_offset_pixels = width / 2
    y_offset_pixels = height / 2

    start_x_units = numpy.random.uniform(-max_x_units, max_x_units, n_points)
    start_y_units = numpy.random.uniform(-max_y_units, max_y_units, n_points)


    # in case seeding is changed
    assert(len(start_x_units) == n_points)

    print "spawning path computations for {} points iterated {} times each".format(n_points,
                                                                                   max_its_per_point)


    # must use threading backend to write to shared memory
    n_jobs = 8
        
    paths = joblib.Parallel(n_jobs = 8, backend="threading")(
        joblib.delayed(compute_and_draw_path)(
            pixels_per_unit,
            start_x_units[i],
            start_y_units[i],
            max_x_units,
            max_y_units,
            max_its_per_point)
        for i in range(n_points))
        
    ts = str(datetime.datetime.now()).split()[1][:10]

    fn_base = 'buddhabrot_{}_points_{}_iterations_{}'.format(n_points, max_its_per_point, ts)

    numpy_fn = fn_base + '.npz'
    if save:
        save_hits(hits, fn_base)
        numpy.savez_compressed(numpy_fn, hits)

    # copy global array
    return numpy_fn, hits.copy()

if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([main, multi_bb])
    parser.dispatch()
                         
