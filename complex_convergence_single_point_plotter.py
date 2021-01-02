import matplotlib, copy, mpmath, multiprocessing, time, math, inspect, shutil, re
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

def sanitize_file_name(s):
    s = re.sub(':', '', s)
    s = re.sub('/', 'div', s)
    s = re.sub('\*', 'mult', s)
    return s

def plot_process(child_conn, initial_value, dps, f_string, stop_iteration_near):
    #Allows plot interactivity while main thread keeps chugging away on function iteration
    points = [initial_value]
    num_point_display_history = [float('inf')]

    # Top level definitions
    baseline_figsize=(20, 10)
    main_plot_left = 0.04
    main_plot_right = 0.99
    main_plot_top = 0.95
    main_plot_bottom = 0.06
    plain_descriptor = f_string + ', beginning from ' + str(initial_value) + ', ' + str(dps) + ' decimal places precision'
    iteration_prefix = 'Iteration {0} of '
    iteration_postfix = ', iteration {0}'
    point_value_string = lambda point: '\n' + mpmath.nstr(point, 30)
    xlabel_1 = 'real value'
    ylabel_1 = 'imaginary value'
    xlabel_0 = 'iterative imaginary values'
    ylabel_0 = 'iteration'
    xlabel_3 = 'iteration'
    ylabel_3 = 'iterative real values'
    label_2 = 'display\nhistory\n(#points)'
    title = iteration_prefix.format(0) + plain_descriptor + point_value_string(initial_value)
    grid_spec_width_ratios = [3, 50]
    grid_spec_height_ratios = [30, 4]
    
    # initialize live displayed plot
    # real monitor = 12.75 in tall x 21.75+1/8 in wide, 1080p, 100dpi plot display seems ok.
    fig = plt.figure(dpi=100, figsize=baseline_figsize)
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=grid_spec_width_ratios, height_ratios=grid_spec_height_ratios)
    gs.update(left=main_plot_left, right=main_plot_right, top=main_plot_top, bottom=main_plot_bottom)

    ax_1 = plt.subplot(gs[1])
    x, y = list(zip(*[(float(mpmath.re(point)), float(mpmath.im(point))) for point in points]))
    plot_1, = plt.plot(x, y, zorder=2,
                       markevery=[-1], marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red')
    plt.axis('scaled')  # force equal axes even on plot window resize
    plt.xlabel(xlabel_1)
    plt.ylabel(ylabel_1)
    plt.title(title)

    ax_0 = plt.subplot(gs[0])
    plot_0, = plt.plot(y, list(range(len(y))), zorder=2)
    plt.xlabel(xlabel_0)
    plt.ylabel(ylabel_0)

    ax_3 = plt.subplot(gs[3])
    plot_3, = plt.plot(list(range(len(x))), x, zorder=2)
    plt.xlabel(xlabel_3)
    plt.ylabel(ylabel_3)

    ax_2 = plt.subplot(gs[2])
    num_point_display_history_box = matplotlib.widgets.TextBox(ax_2, label_2, initial='inf')
    def num_point_display_history_box_submit(text):
        if text == 'inf':
            num_point_display_history[0] = float('inf')
            num_point_display_history_box_save.set_val(text)
        else:
            try:
                parsed_text = int(text)
                assert parsed_text > 0
                num_point_display_history[0] = parsed_text
                num_point_display_history_box_save.set_val(text)
            except:
                print('unacceptable num_point_display_history entry "' + text + '"')
    num_point_display_history_box.on_submit(num_point_display_history_box_submit)


    plt.show(block=False)

    # initialize saved plot
    fig_save = plt.figure(dpi=100, figsize=baseline_figsize)
    gs_save = matplotlib.gridspec.GridSpec(2, 2, width_ratios=grid_spec_width_ratios, height_ratios=grid_spec_height_ratios)
    gs_save.update(left=main_plot_left, right=main_plot_right, top=main_plot_top, bottom=main_plot_bottom)

    ax_1_save = plt.subplot(gs_save[1])
    x_save, y_save = list(zip(*[(float(mpmath.re(point)), float(mpmath.im(point))) for point in points]))
    plot_1_save, = plt.plot(x_save, y_save, zorder=2)
    plt.axis('scaled')  # force equal axes even on plot window resize
    plt.xlabel(xlabel_1)
    plt.ylabel(ylabel_1)
    plt.title(title)

    ax_0_save = plt.subplot(gs_save[0])
    plot_0_save, = plt.plot(y_save, list(range(len(y_save))), zorder=2)
    plt.xlabel(xlabel_0)
    plt.ylabel(ylabel_0)

    ax_3_save = plt.subplot(gs_save[3])
    plot_3_save, = plt.plot(list(range(len(x_save))), x_save, zorder=2)
    plt.xlabel(xlabel_3)
    plt.ylabel(ylabel_3)

    ax_2_save = plt.subplot(gs_save[2])
    num_point_display_history_box_save = matplotlib.widgets.TextBox(ax_2_save, label_2, initial='inf')

    # configure saved image filenames
    save_name_string_addition = ', stop_iteration_near ' + str(stop_iteration_near)
    sanitized_save_name = sanitize_file_name(plain_descriptor) + save_name_string_addition
    save_name_iteration_base = 'points/' + sanitized_save_name + iteration_postfix + '.png'
    save_name_final          = 'points/Final ' + sanitized_save_name + '.png'

    child_conn.send(True)  # signal main thread ready for a new color map

    while(1):
        if child_conn.poll():
            # collect new data
            point, iteration = child_conn.recv()
            points.append(point)

            if num_point_display_history[0] == float('inf'):
                displayed_points = points
            else:
                displayed_points = points[-num_point_display_history[0]:]

            # update displayed figure
            x, y = list(zip(*[(float(mpmath.re(point)), float(mpmath.im(point))) for point in displayed_points]))
            ax_1.set_title(iteration_prefix.format(iteration) + plain_descriptor + point_value_string(point))

            plot_0.set_data(y, list(range(len(y))))
            plot_1.set_data(x, y)
            plot_3.set_data(list(range(len(x))), x, )
            for ax in [ax_0, ax_1, ax_3]:
                ax.relim()
                ax.autoscale()
            fig.canvas.draw()
            fig.canvas.draw()  # intentionally repeated
            fig.canvas.flush_events()

            # save figure on each iteration
            x_save, y_save = list(zip(*[(float(mpmath.re(point)), float(mpmath.im(point))) for point in displayed_points]))
            ax_1_save.set_title(iteration_prefix.format(iteration) + plain_descriptor + point_value_string(point))
            plot_0_save.set_data(y_save, list(range(len(y_save))))
            plot_1_save.set_data(x_save, y_save)
            plot_3_save.set_data(list(range(len(x_save))), x_save, )
            for ax in [ax_0_save, ax_1_save, ax_3_save]:
                ax.relim()
                ax.autoscale()
            fig_save.canvas.draw()
            fig_save.canvas.draw()  # intentionally repeated
            fig_save.canvas.flush_events()
            fig_save.savefig(save_name_final, dpi=100, bbox_inches='tight')
            shutil.copyfile(save_name_final, save_name_iteration_base.format(iteration))
            
            # signal main thread ready for a new color map
            child_conn.send(True)
        else:
            background_plt_pause(0.01)


if __name__ == '__main__':
    # set precision
    mpmath.mp.dps=1000

    # define iteration function
    # f = lambda z: mpmath.exp(-mpmath.power(z, 2))
    # f = lambda z: mpmath.power(z, mpmath.mpc('0.5','0'))
    # f = lambda z: z - (z - z / mpmath.fabs(z)) * mpmath.mpc('0.99','0')
    # f = lambda z: mpmath.mpc('0.5','0') * mpmath.sin(mpmath.mpc('1','0') / z)
    # f = lambda z: mpmath.mpc('1.0','0') / z
    # f = lambda z: mpmath.exp(mpmath.mpc('-1.0','0.0') / z) - mpmath.exp(mpmath.mpc('1.0','0.0') / z)
    # f = lambda z: mpmath.power(mpmath.mpc('0.5', '0'), z)
    # f = lambda z: mpmath.sin(mpmath.mpc('1','0') / (z + mpmath.mpc('0','0.27')))
    # f = lambda z: mpmath.sin(mpmath.mpc('1','0') / (z + mpmath.mpc('0','2.5')))
    # f = lambda z: z + mpmath.mpc('5', '0') * mpmath.sin(mpmath.re(z)) + mpmath.mpc('0', '5') * mpmath.sin(mpmath.im(z))
    f = lambda z: mpmath.power(z, z)
    
    # stop_iteration_near = [mpmath.mpc('0','-2.5')]
    stop_iteration_near = [0]

    # point = mpmath.mpc('-1.34','-2.0')
    # point = mpmath.mpc('0.15530079','0.0')
    # point = mpmath.mpc('0.3','3.0')
    point = mpmath.mpc('2.8834','1.2230')

    # setup interactive plot
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=plot_process, args=(child_conn,
                                                           point,
                                                           mpmath.mp.dps,
                                                           inspect.getsource(f).strip(),
                                                           stop_iteration_near,
                                                          ))
    p.start()

    def reraise_error():
        print('Please wait... Generating error message...')
        print('point', point)
        raise

    iteration = 0
    while(1):
        print('iteration', iteration, 'value', mpmath.nstr(point, 30), end='\n')
        # print('iteration', iteration, end='\r')
        iteration += 1

        # update value
        # don't reprocess errors
        if point == mpmath.nan:
            raise ValueError('Got a NaN!')

        # stop iteration on values sufficiently close to say zero
        if any([mpmath.almosteq(stop_iteration_value, point) for stop_iteration_value in stop_iteration_near]):
            point == mpmath.nan

        # try processing point
        try:
            new_point = f(point)
        except OverflowError as e:
            if e.__str__() == 'int too large to convert to float':
                reraise_error()
            else:
                reraise_error()
        # except:  # handle unknown errors
        #     values[i] == mpmath.nan
        #     colors[i] = (1.0, 0.0, 0.0)
        #     reraise_error()
        else: # color point according to normal math rules and store new value
            point = new_point

        # plot latest data
        parent_conn.recv()  # block until we can send new plot data
        parent_conn.send((point, iteration))

