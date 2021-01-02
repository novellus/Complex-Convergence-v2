import matplotlib, copy, mpmath, multiprocessing, time, math, inspect, shutil, re, traceback, os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from func_timeout import func_timeout, FunctionTimedOut, StoppableThread


###################################
def background_plt_pause(interval):
    """
    Blatantly stolen from matplotlib source, show line removed to update in background.
    Per guidance at https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7

    Pause for *interval* seconds.

    If there is an active figure, it will be updated and displayed before the
    pause, and the GUI event loop (if any) will run during the pause.

    This can be used for crude animation.  For more complex animation, see
    :mod:`matplotlib.animation`.

    Notes
    -----
    This function is experimental; its behavior may be changed or extended in a
    future release.
    """
    manager = matplotlib._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        # show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)
###################################


###################################
# Blatantly stolen from func_timeout source
# Modify func_timeout so that it doesn't try to turn the args into strings everytime it triggers
# this application has very large numerical args which can take too long to compute string representations

import threading
import sys

def raise_exception(exception):
    raise exception[0] from None

def func_timeout(timeout, func, args=(), kwargs=None):
    '''
        func_timeout - Runs the given function for up to #timeout# seconds.
        Raises any exceptions #func# would raise, returns what #func# would return (unless timeout is exceeded), in which case it raises FunctionTimedOut
        @param timeout <float> - Maximum number of seconds to run #func# before terminating
        @param func <function> - The function to call
        @param args    <tuple> - Any ordered arguments to pass to the function
        @param kwargs  <dict/None> - Keyword arguments to pass to the function.
        @raises - FunctionTimedOut if #timeout# is exceeded, otherwise anything #func# could raise will be raised
        If the timeout is exceeded, FunctionTimedOut will be raised within the context of the called function every two seconds until it terminates,
        but will not block the calling thread (a new thread will be created to perform the join). If possible, you should try/except FunctionTimedOut
        to return cleanly, but in most cases it will 'just work'.
        @return - The return value that #func# gives
    '''

    if not kwargs:
        kwargs = {}
    if not args:
        args = ()

    ret = []
    exception = []
    isStopped = False

    def funcwrap(args2, kwargs2):
        try:
            ret.append( func(*args2, **kwargs2) )
        except FunctionTimedOut:
            # Don't print traceback to stderr if we time out
            pass
        except Exception as e:
            exc_info = sys.exc_info()
            if isStopped is False:
                # Assemble the alternate traceback, excluding this function
                #  from the trace (by going to next frame)
                # Pytohn3 reads native from __traceback__,
                # python2 has a different form for "raise"
                e.__traceback__ = exc_info[2].tb_next
                exception.append( e )

    thread = StoppableThread(target=funcwrap, args=(args, kwargs))
    thread.daemon = True

    thread.start()
    thread.join(timeout)

    stopException = None
    if thread.is_alive():
        isStopped = True

        class FunctionTimedOutTempType(FunctionTimedOut):
            def __init__(self):
                return FunctionTimedOut.__init__(self, '', timeout, func)

        FunctionTimedOutTemp = type('FunctionTimedOut' + str( hash( "%d_%d" %(id(timeout), id(func))) ), FunctionTimedOutTempType.__bases__, dict(FunctionTimedOutTempType.__dict__))

        stopException = FunctionTimedOutTemp
        thread._stopThread(stopException)
        thread.join(min(.1, timeout / 50.0))
        raise FunctionTimedOut('', timeout, func)
    else:
        # We can still cleanup the thread here..
        # Still give a timeout... just... cuz..
        thread.join(.5)

    if exception:
        raise_exception(exception)

    if ret:
        return ret[0]
###################################


def sanitize_file_name(s):
    s = re.sub(':', '', s)
    s = re.sub('/', 'div', s)
    s = re.sub('\*', 'mult', s)
    return s


def plot_process(child_conn, colors, num_x_points, num_y_points, extent, dps, compute_timeout):
    #Allows plot interactivity while main thread keeps chugging away on function iteration
    
    # Top level definitions
    baseline_figsize=(20, 10)
    main_plot_left = 0.04
    main_plot_right = 0.99
    main_plot_top = 0.975
    main_plot_bottom = 0.06
    f_string = inspect.getsource(f).strip()
    plain_descriptor = f_string + ', ' + str(dps) + ' decimal places precision'
    plain_descriptor_short = f_string + ',' + str(dps) + 'dpp'
    predefined_zones_of_attraction_descriptor = ', ' + inspect.getsource(predefined_zones_of_attraction[0][0]).strip() if predefined_zones_of_attraction else []  # TODO, poor implementation
    iteration_prefix = 'Iteration {0} of '
    iteration_postfix = ',I#{0}'
    xlabel = 'real initial value, ' + str(num_x_points) + ' columns'
    ylabel = 'imaginary initial value, ' + str(num_y_points) + ' rows'
    title = iteration_prefix.format(0) + plain_descriptor + predefined_zones_of_attraction_descriptor
    baground_color = '#E0E0E0'  # set to be different from all valid plot colors
    legend_convergence_patch = lambda converged_color_descriptor: matplotlib.patches.Patch(color='black', label='converged within mpmath.almosteq(new_val, val)' + converged_color_descriptor)
    legend_params = {
        'handles': [
                 legend_convergence_patch(''), 
                 matplotlib.patches.Patch(color='#0000FF', label='compute timedout (' + str(compute_timeout) + 's) or overflow/memory/recursion error'),
                 matplotlib.patches.Patch(color='#42d4ff', label='stopped iteration on mpmath.almosteq(x, val) for x in ' + str(stop_iteration_near)),
                 # matplotlib.patches.Patch(color='red', label='unknown error'),
                 matplotlib.patches.Patch(color='white', label='not (yet) converged or otherwise'),
                ],
        'fontsize': 'xx-small',
        'ncol': 4,
        'loc': 3,
        'bbox_to_anchor': [0, 0],
        'facecolor': baground_color,
    }
    
    # initialize live displayed plot
    # real monitor = 12.75 in tall x 21.75+1/8 in wide, 1080p, 100dpi plot display seems ok.
    fig = plt.figure(dpi=100, figsize=baseline_figsize, facecolor=baground_color)
    gs = matplotlib.gridspec.GridSpec(1, 1)
    gs.update(left=main_plot_left, right=main_plot_right, top=main_plot_top, bottom=main_plot_bottom)
    ax = plt.subplot(gs[0])
    Z = np.array(colors).reshape(num_y_points, num_x_points, 3)
    im = plt.imshow(Z, origin='low', aspect='auto', interpolation=None, extent=extent, zorder=2)
    plt.axis('scaled')  # force equal axes even on plot window resize
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_transform=fig.transFigure, **legend_params)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show(block=False)

    # initialize saved plot, at minimum resolution that captures all image data
    # make small adjustments to image size to equalize real-data resolution of displayed image axes 
    #   this is safe with respect to statically sized plot objects, since we initially specify plot domain such as to
    #   guestimate point densities on each axis to get us close with other plot elements.
    min_save_fig_size_x = baseline_figsize[0]
    min_save_fig_size_y = baseline_figsize[1]
    min_dpi_x = float(num_x_points) / (min_save_fig_size_x * (main_plot_right - main_plot_left))
    min_dpi_y = float(num_y_points) / (min_save_fig_size_y * (main_plot_top - main_plot_bottom))
    if min_dpi_x <= min_dpi_y:
        save_fig_size_x = min_save_fig_size_x
        save_fig_size_y = ((min_save_fig_size_x * (main_plot_right - main_plot_left)) / float(num_x_points) * float(num_y_points)) / (main_plot_top - main_plot_bottom)
        save_dpi = min_dpi_x
    else:
        save_fig_size_y = min_save_fig_size_y
        save_fig_size_x = ((min_save_fig_size_y * (main_plot_top - main_plot_bottom)) / float(num_y_points) * float(num_x_points)) / (main_plot_right - main_plot_left)
        save_dpi = min_dpi_y
    # raise dpi up to at least 100 for visibility of title, key, etc, but in an even multiple of intended dpi
    if save_dpi < 100:
        save_dpi *= int(math.ceil(100/save_dpi))

    fig_save = plt.figure(dpi=save_dpi, figsize=(save_fig_size_x, save_fig_size_y), facecolor=baground_color)
    gs_save = matplotlib.gridspec.GridSpec(1, 1)
    gs_save.update(left=main_plot_left, right=main_plot_right, top=main_plot_top, bottom=main_plot_bottom)
    ax_save = plt.subplot(gs_save[0])
    Z_save = np.array(colors).reshape(num_y_points, num_x_points, 3)
    im_save = plt.imshow(Z, origin='low', aspect='auto', interpolation=None, extent=extent, zorder=2)
    plt.axis('scaled')  # force equal axes even on plot window resize
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_transform=fig_save.transFigure, **legend_params)
    ax_save.spines['top'].set_visible(False)
    ax_save.spines['right'].set_visible(False)
    ax_save.spines['bottom'].set_visible(False)
    ax_save.spines['left'].set_visible(False)

    # configure saved image filenames
    save_name_string_addition = ',' + str(num_x_points) + 'c+' + str(num_y_points) + 'r,extent ' + str(extent) + ',sin' + str(stop_iteration_near)
    sanitized_save_name = sanitize_file_name(plain_descriptor) + save_name_string_addition
    save_name_iteration_base = os.path.join('images', sanitized_save_name + iteration_postfix + '.png')
    save_name_final          = os.path.join('images', 'Final ' + sanitized_save_name + '.png')

    child_conn.send(True)  # signal main thread ready for a new color map

    while(1):
        if child_conn.poll():
            # collect new data
            colors, iteration, converged_color_descriptor = child_conn.recv()
            legend_params['handles'][0] = legend_convergence_patch(converged_color_descriptor)
            
            # update displayed figure
            Z = np.array(colors).reshape(num_y_points, num_x_points, 3)
            im.set_data(Z)
            ax.set_title(iteration_prefix.format(iteration) + plain_descriptor + predefined_zones_of_attraction_descriptor)
            ax.legend(bbox_transform=fig.transFigure, **legend_params)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # save figure on each iteration
            Z_save = np.array(colors).reshape(num_y_points, num_x_points, 3)
            im_save.set_data(Z_save)
            ax_save.set_title(iteration_prefix.format(iteration) + plain_descriptor + predefined_zones_of_attraction_descriptor)
            ax_save.legend(bbox_transform=fig_save.transFigure, **legend_params)
            fig_save.canvas.draw()
            fig_save.canvas.flush_events()
            fig_save.savefig(save_name_final, dpi=save_dpi, bbox_inches='tight', facecolor=baground_color)
            shutil.copyfile(save_name_final, save_name_iteration_base.format(iteration))
            
            # signal main thread ready for a new color map
            child_conn.send(True)
        else:
            background_plt_pause(0.1)


def calc_points_threaded(child_conn, values, colors, convergence_iterations, dps, compute_timeout):
    num_points = len(values)
    mpmath.mp.dps = dps

    iteration = 0
    while 1:
        iteration += 1

        # update values and colors matrices
        try:
            values, colors, convergence_iterations = func_timeout(compute_timeout * num_points, compute_points_wholesale, args=(values, colors, convergence_iterations, iteration, stop_iteration_near, predefined_zones_of_attraction, f))
        except FunctionTimedOut:
            values, colors, convergence_iterations = compute_points_individually(values, colors, convergence_iterations, iteration, stop_iteration_near, predefined_zones_of_attraction, f, compute_timeout)

        child_conn.send((values, colors, convergence_iterations))
        child_conn.recv()  # block until parent receives data


def compute_points_wholesale(values, colors, convergence_iterations, iteration, stop_iteration_near, predefined_zones_of_attraction, f):
    new_values = ['fuck']*len(values)
    new_colors = ['fuck']*len(colors)
    new_convergence_iterations = ['fuck']*len(convergence_iterations)
    # update values and colors matrices
    for i, val in enumerate(values):
        # don't reprocess errors
        if mpmath.isnan(val):
            new_values[i] = val
            new_colors[i] = colors[i]
            continue

        # stop iteration on values sufficiently close to, for instance zero
        if any([mpmath.almosteq(stop_iteration_value, val) for stop_iteration_value in stop_iteration_near]):
            new_values[i] = mpmath.nan
            new_colors[i] = (0.2588, 0.8314, 1.0)
            continue

        # sometimes convergence is too slow to measure with mathematical precision alone, so we predefine some zones of attraction where convergence is known
        # some examples are converging at a rate that slows the closer it is, or a decaying swirl that traces an infinite path length around the convergence point (but which still converges at infinity)
        satisfied_zone_of_attraction = False
        for condition, convergence_value in predefined_zones_of_attraction:
            if condition(val):
                new_values[i] = convergence_value
                new_colors[i] = 'converged'
                if colors[i] != 'converged':
                    new_convergence_iterations[i] = iteration
                else:
                    new_convergence_iterations[i] = convergence_iterations[i]
                satisfied_zone_of_attraction = True
                break
        if satisfied_zone_of_attraction == True:
            continue

        # try processing point
        try:
            new_val = f(val)
        except FunctionTimedOut:
                raise  # just don't let this get caught by the unknown exception clause below, I don't need to see an error message
        except OverflowError as e:
            if e.__str__() in ['int too large to convert to float',
                               'Python int too large to convert to C ssize_t',
                               'cannot convert float infinity to integer',
                               'too many digits in integer']:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
            else:
                reraise_error(val)
        except MemoryError:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
        except RecursionError:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
        except:  # handle unknown errors
            reraise_error(val)
        else: # color point according to normal math rules and store new value
            if mpmath.almosteq(new_val, val):
                new_colors[i] = 'converged'
                if colors[i] != 'converged':
                    new_convergence_iterations[i] = iteration
                else:
                    new_convergence_iterations[i] = convergence_iterations[i]
            else:
                new_colors[i] = (1.0, 1.0, 1.0)
            new_values[i] = new_val

    return new_values, new_colors, new_convergence_iterations


def compute_points_individually(values, colors, convergence_iterations, iteration, stop_iteration_near, predefined_zones_of_attraction, f, compute_timeout):
    new_values = ['fuck']*len(values)
    new_colors = ['fuck']*len(colors)
    new_convergence_iterations = ['fuck']*len(convergence_iterations)
    # update values and colors matrices
    for i, val in enumerate(values):
        # don't reprocess errors
        if mpmath.isnan(val):
            new_values[i] = val
            new_colors[i] = colors[i]
            continue

        # stop iteration on values sufficiently close to, for instance zero
        if any([mpmath.almosteq(stop_iteration_value, val) for stop_iteration_value in stop_iteration_near]):
            new_values[i] = mpmath.nan
            new_colors[i] = (0.2588, 0.8314, 1.0)
            continue

        # sometimes convergence is too slow to measure with mathematical precision alone, so we predefine some zones of attraction where convergence is known
        # some examples are converging at a rate that slows the closer it is, or a decaying swirl that traces an infinite path length around the convergence point (but which still converges at infinity)
        satisfied_zone_of_attraction = False
        for condition, convergence_value in predefined_zones_of_attraction:
            if condition(val):
                new_values[i] = convergence_value
                new_colors[i] = 'converged'
                if colors[i] != 'converged':
                    new_convergence_iterations[i] = iteration
                else:
                    new_convergence_iterations[i] = convergence_iterations[i]
                satisfied_zone_of_attraction = True
                break
        if satisfied_zone_of_attraction == True:
            continue

        # try processing point
        try:
            new_val = func_timeout(compute_timeout, f, args=(val,))
        except FunctionTimedOut:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
        except OverflowError as e:
            if e.__str__() in ['int too large to convert to float',
                               'Python int too large to convert to C ssize_t',
                               'cannot convert float infinity to integer',
                               'too many digits in integer']:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
            else:
                reraise_error(val)
        except MemoryError:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
        except RecursionError:
                new_values[i] = mpmath.nan
                new_colors[i] = (0.0, 0.0, 1.0)
        except:  # handle unknown errors
            reraise_error(val)
        else: # color point according to normal math rules and store new value
            if mpmath.almosteq(new_val, val):
                new_colors[i] = 'converged'
                if colors[i] != 'converged':
                    new_convergence_iterations[i] = iteration
                else:
                    new_convergence_iterations[i] = convergence_iterations[i]
            else:
                new_colors[i] = (1.0, 1.0, 1.0)
            new_values[i] = new_val

    return new_values, new_colors, new_convergence_iterations

def reraise_error(val):
    print()
    print('Please wait... Generating error message...')
    print('\tval', val)
    traceback.print_exc()
    raise


# define iteration function
# f = lambda z: mpmath.exp(-mpmath.power(z, 2))
# f = lambda z: mpmath.power(z, mpmath.mpc('0.5','0'))
# f = lambda z: z - (z - z / mpmath.fabs(z)) * mpmath.mpc('0.99','0')
# f = lambda z: mpmath.sin(mpmath.mpc('2.5','0') * mpmath.pi / z)
# f = lambda z: mpmath.sin(mpmath.mpc('7.8','0') / z)
# f = lambda z: mpmath.sin(mpmath.mpc('1','0') / z)
# f = lambda z: mpmath.sin(mpmath.mpc('1','0') / (z + mpmath.mpc('0','1.5')))
# f = lambda z: mpmath.sin(mpmath.mpc('1','0') / (z + mpmath.mpc('0','2.6')))
# f = lambda z: mpmath.power(mpmath.mpc('0.5', '0'), z)
# f = lambda z: z + mpmath.sin(z)
# f = lambda z: z + mpmath.mpc('1.5', '0') * mpmath.sin(z)
f = lambda z: mpmath.power(z, z)

# stop_iteration_near = [mpmath.mpc('0','-1.5')]
# stop_iteration_near = [0]
stop_iteration_near = []

# [(comdition_func, converged_result), ...]
predefined_zones_of_attraction = [(lambda p: mpmath.mpf('0.5') < mpmath.re(p) <= mpmath.mpf('1') and mpmath.mpf('-0.5') < mpmath.im(p) < mpmath.mpf('0.5'), mpmath.mpc('1', '0'))]


if __name__ == '__main__':
    # set precision
    mpmath.mp.dps = 50
    max_legend_printed_dps = min(mpmath.mp.dps, 30)

    compute_timeout = 0.001

    # Total thread count = this thread + plotting thread + compute threads
    # num_compute_threads = 1
    num_compute_threads = 25

    # magic number, keep an aspect ratio that satisifies an equivilent dpi on each axis of final image
    # This doesn't need to be exact (hence why we allow small adjustments in point counts) since the plot function
    #   will make small adjustments to image size to match up each axes requested dpi's required to capture all data.
    # It does need to be close tho, since letting the plotter make large changes in image size will make 
    #   statically sized plot elements (eg fonts) do undesirable things (eg title larger than image or smaller than dpi).
    x_y_plot_size_ratio = 2.07725  # empirically designed
    
    # define graph domain
    num_y_points = 400  # must be even to avoid zero when zero is not in domain?
    num_x_points = int(num_y_points * x_y_plot_size_ratio)  # set aspect ratio
    if not num_y_points%2:  # keep x odd/even parity with y
        num_x_points = math.floor(num_x_points / 2) * 2
    y_min = -3
    y_max = 3
    x_min = 0 - float(num_x_points) / num_y_points * (y_max - y_min) / 2  # upper and max x-bounds are computed, and centered on the leading number in this equation
    # x_min = -20
    x_max = x_min + float(num_x_points) / num_y_points * (y_max - y_min)  # keep the axes at equal display ratios
    print('num_x_points', num_x_points)
    print('num_y_points', num_y_points)
    print('x range', x_min, x_max)
    print('y range', y_min, y_max)

    print('Building initial data set')
    points = [mpmath.mpc(x, y) \
              for y in mpmath.linspace(y_min, y_max, num_y_points) \
              for x in mpmath.linspace(x_min, x_max, num_x_points)]

    values = copy.deepcopy(points)  # each point is its own initial condition, by design intent
    colors = [(1.0, 1.0, 1.0)] * len(values)
    convergence_iterations = [None] * len(values)


    # setup interactive plot
    parent_conn_plot, child_conn_plot = multiprocessing.Pipe()
    p = multiprocessing.Process(target=plot_process, args=(child_conn_plot,
                                                           colors,
                                                           num_x_points,
                                                           num_y_points,
                                                           [x_min, x_max, y_min, y_max],
                                                           mpmath.mp.dps,
                                                           compute_timeout,
                                                          ))
    p.start()



    # setup calculation threads
    # detemine number of points in each thread
    num_points = len(points)
    num_compute_threads = min(num_compute_threads, num_points)  # don't create empty threads
    min_num_points_per_thread = int(math.floor(num_points / num_compute_threads))  # every thread gets atleast this many points
    num_indivisible_points = num_points - min_num_points_per_thread * num_compute_threads  # distribute one extra point to the last this many threads
    num_threads_without_extra_points = num_compute_threads - num_indivisible_points

    # now actually create the threads
    compute_parent_conns = []
    i_values = 0
    for i_thread in range(num_compute_threads):
        print('creating thread', i_thread + 1, end='\r')
        parent_conn, child_conn = multiprocessing.Pipe()
        compute_parent_conns.append(parent_conn)

        num_points_this_thread = min_num_points_per_thread + (0 if i_thread < num_threads_without_extra_points else 1)

        values_child = values[i_values: i_values + num_points_this_thread]
        colors_child = colors[i_values: i_values + num_points_this_thread]
        convergence_iterations_child = convergence_iterations[i_values: i_values + num_points_this_thread]
        i_values += num_points_this_thread

        p = multiprocessing.Process(target=calc_points_threaded, args=(child_conn, values_child, colors_child, convergence_iterations_child, mpmath.mp.dps, compute_timeout))
        p.start()

    print()
    iteration = 0
    while(1):
        print('iteration', iteration, end='\r')
        iteration += 1

        i = 0
        # update values and colors matrices
        for parent_conn in compute_parent_conns:
            values_subset, colors_subset, convergence_iterations_subset = parent_conn.recv()  # block until each compute thread finishes
            parent_conn.send(True)  # indicate compute thread can proceed with next computation
            values[i: i + len(values_subset)] = values_subset
            colors[i: i + len(colors_subset)] = colors_subset
            convergence_iterations[i: i + len(convergence_iterations_subset)] = convergence_iterations_subset
            i += len(values_subset)

        # scale converged colors
        convergence_x_values = []
        convergence_y_values = []
        convergence_iteration_values = []
        for i, val in enumerate(values):
            if colors[i] == 'converged':
                convergence_x_values.append(mpmath.re(val))
                convergence_y_values.append(mpmath.im(val))
                convergence_iteration_values.append(convergence_iterations[i])
        if len(convergence_x_values) == 0:
            converged_color_descriptor = ''
        if len(convergence_x_values) == 1:
            for i, val in enumerate(values):
                if colors[i] == 'converged':
                    colors[i] = (0.0, 0.0, 0.0)
            converged_color_descriptor = ''
        elif len(convergence_x_values) > 1:
            converged_x_max = max(convergence_x_values)
            converged_x_min = min(convergence_x_values)
            converged_y_max = max(convergence_y_values)
            converged_y_min = min(convergence_y_values)
            converged_iteration_max = max(convergence_iteration_values)
            converged_iteration_min = min(convergence_iteration_values)

            if mpmath.almosteq(converged_x_min, converged_x_max):
                red_descriptor = ' - Red: all real(converged value) within tolerance of {0} and scaled to 1'
                red_function = lambda val: 1.0
            else:
                red_descriptor = ' - Red: real(converged value) scaled from [{0}, {1}]'
                red_function = lambda val: float((mpmath.re(val) - converged_x_min) / (converged_x_max - converged_x_min))

            if mpmath.almosteq(converged_y_min, converged_y_max):
                green_descriptor = ', Green: all imag(converged value) within tolerance of {0} and scaled to 1'
                green_function = lambda val: 1.0
            else:
                green_descriptor = ', Green: imag(converged value) scaled from [{0}, {1}]'
                green_function = lambda val: float((mpmath.im(val) - converged_y_min) / (converged_y_max - converged_y_min))

            blue_max_scale = 0.75  # upper values reserved for error codes, and kept visually distinct from convergent values
            if converged_iteration_min == converged_iteration_max:
               blue_descriptor = ',blue: all convergence iteration within tolerance of {0} and scaled to {2}'
               blue_function = lambda val: blue_max_scale
            else:
               blue_descriptor = ',blue: convergence iteration scaled from [{0}, {1}] to [0, {2}]'
               blue_function = lambda i: float(convergence_iterations[i] - converged_iteration_min) / (converged_iteration_max - converged_iteration_min) * blue_max_scale

            converged_color_descriptor = red_descriptor.format(mpmath.nstr(converged_x_min, max_legend_printed_dps), mpmath.nstr(converged_x_max, max_legend_printed_dps)) + \
                                         green_descriptor.format(mpmath.nstr(converged_y_min, max_legend_printed_dps), mpmath.nstr(converged_y_max, max_legend_printed_dps)) + \
                                         blue_descriptor.format(converged_iteration_min, converged_iteration_max, blue_max_scale)
            for i, val in enumerate(values):
                if colors[i] == 'converged':
                    r = red_function(val)
                    g = green_function(val)
                    b = blue_function(i)
                    colors[i] = (r, g, b)

        # plot latest data
        parent_conn_plot.recv()  # block until we can send new plot data
        parent_conn_plot.send((colors, iteration, converged_color_descriptor))

