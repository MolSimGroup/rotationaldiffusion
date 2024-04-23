import argparse

def parse_args(*args_to_parse):
    parser = argparse.ArgumentParser(description="""Compute the rotational diffusion constants of a molecule from 
    the output of 'gmx rotmat', i.e., from rotation matrices extracted from MD simulations.
    
    Follows the procedure described in Chen et. al. [1]: 
    0. Load the rotational matrices describing rotations from trajectory frame to reference frame.
    1. Convert the rotation matrices to quaternions q(t). Invert the direction of rotation (now reference -> 
    trajectory).
    2. Compute the quaternions q_tau(tau) = <q^{-1}(t) * q(t+tau)> (tau), which describe the rotation from the
    trajectory frame at time t to the trajectory frame at time t+tau, averaged over all starting times t.
    """)

    parser._optionals.title = 'General options'
    parser.add_argument('--do-iso', action='store_true', dest='do_iso',
                        help='compute isotropic diffusion')
    parser.add_argument('--do-aniso', action='store_true', dest='do_aniso',
                        help='compute anisotropic diffusion')
    parser.add_argument('--do-full-tensor', action='store_true',
                        dest='do_full_tensor',
                        help='return full tensor T_ij in PAF')
    parser.add_argument('-ait', action='store_true',
                        dest='analyze_individual_trajectories',
                        help='analyze individual trajectories (false)')

    # TODO: implement option for loading .npy arrays
    parser_4loading = parser.add_argument_group("Options for loading data")
    # parser_4loading_files = parser_4loading.add_mutually_exclusive_group(required=True)
    parser_4loading.add_argument('-f', action='extend', nargs='+', type=Path,
                                 dest='files', required=True,
                                 help="'gmx rotmat' output files OR numpy .npy files")
    # parser_4loading_files.add_argument('-nf', action='extend', nargs='+', type=Path, dest='files',
    #                              help=".npy file")
    parser_4loading.add_argument('-tu', action='store', type=str, dest='unit',
                                 default='ps',
                                 help='unit for time values (ps)')
    parser_4loading.add_argument('-b', action='store', type=float,
                                 dest='begin', default=0,
                                 help='first time to use in UNIT (0)')
    parser_4loading.add_argument('-e', action='store', type=float, dest='end',
                                 default=-1,
                                 help='last time to use in UNIT (-1)')
    parser_4loading.add_argument('-s', action='store', type=int, dest='skip',
                                 default=1,
                                 help='use every SKIP-th frame (1)')

    parser_4output = parser.add_argument_group("Options for output")
    parser_4output.add_argument('--out-file', action='store', type=Path,
                                dest='out_file',
                                help='write all results as pickled dictionary to disk')
    parser_4output.add_argument('--print-taus', action='store_true',
                                dest='print_taus',
                                help='print summary of computed tau_corr')
    parser_4output.add_argument('--print-D', action='store_true',
                                dest='print_D',
                                help='print summary of computed Ds')
    parser_4output.add_argument('-ow', action='store_true', dest='overwrite',
                                help='allow overwriting OUT_FILE')

    # TODO: implement allowing multiple fit times
    # TODO: implement option to choose fitting a in exponential
    parser_4correlation = parser.add_argument_group(
        'Options for correlation fct. q_corr(tau)')
    parser_4correlation.add_argument('-ts', action='store', type=float,
                                     dest='tau_step', default=-1,
                                     help='spacing of discrete lag time tau in UNIT (-1)')
    parser_4correlation.add_argument('-tm', action='store', type=float,
                                     dest='tau_max', default=-1,
                                     help='maximum lag time tau in UNIT (-1)')
    parser_4correlation.add_argument('-pt', action='extend', nargs='+',
                                     type=float, dest='PAF_times',
                                     default=[-1],
                                     help='lag times tau in UNIT to compute PAFs (-1)')

    parser_4fitting = parser.add_argument_group('Options for fitting')
    parser_4fitting_tmf = parser_4fitting.add_mutually_exclusive_group(
        required=True)
    parser_4fitting_tmf.add_argument('-tmf', action='extend', nargs='+',
                                     type=float, dest='tau_max_4fitting',
                                     help='maximum lag time tau in UNIT to consider for fitting')
    parser_4fitting_tmf.add_argument('-sss', action='extend', nargs=3,
                                     type=float, dest='SSS',
                                     help='start, stop, and skip values to be used in np.arange for defining multiple maximum lag times tau for fitting in UNIT at once')
    parser_4fitting.add_argument('-fc', action='store_true', dest='fitC',
                                 help='use flexible variable C for fitting')

    parser_4uncertainty = parser.add_argument_group(
        'Options for estimating uncertainties')
    parser_4uncertainty.add_argument('-nc', action='store', type=int,
                                     dest='n_chunks', default=0,
                                     help='number of chunks (0)')

    if not args_to_parse:
        args = parser.parse_args()
    else:
        args = parser.parse_args(*args_to_parse)

    # Check if input files exist.
    for file in args.files:
        if not file.is_file():
            parser.error(f"argument -f: file {file} does not exist")

    # Verify that output file does not exist.
    if args.out_file and args.out_file.exists() and not args.overwrite:
        parser.error(
            f"argument --out-file: file {args.out_file} already exists")

    # Get time information from first file.
    try:
        time = np.loadtxt(args.files[0], comments=['@', '#'])[:, 0]
    except ValueError:
        parser.error(
            f"argument -f: cannot load data from file {args.files[0]} into numpy array")
    time_fin = time[-1] - time[0]
    time_step_data = time[1] - time[0]
    time_step_skip = time_step_data * args.skip

    # Check 'begin' and 'end'.
    if args.end == -1:
        args.end = time_fin
    if args.begin < 0:
        parser.error(
            f"argument -b: must be >= 0 {args.unit} (is {args.begin:.1f} {args.unit})")
    elif args.end <= args.begin:
        parser.error(
            f"argument -e: must be > {args.begin} {args.unit} (is {args.end:.1f} {args.unit})")
    elif args.end > time_fin:
        parser.error(
            f"argument -e: must be <= {time_fin} {args.unit} (is {args.end:.1f} {args.unit})")
    elif args.begin % time_step_data:
        parser.error(
            f"argument -b: must be a multiple of {time_step_data} {args.unit} (is {args.begin:.1f} {args.unit})")
    elif args.end % time_step_data:
        parser.error(
            f"argument -e: must be a multiple of {time_step_data} {args.unit} (is {args.end:.1f} {args.unit})")
    elif (args.end - args.begin) % time_step_skip:
        parser.error(
            f"arguments -e and -b: END-BEGIN must be a multiple of {time_step_skip} {args.unit} (is {args.end - args.begin} {args.unit})")

    # Check 'tau_step'.
    if args.tau_step == -1:
        args.tau_step = time_step_skip
    elif args.tau_step % time_step_skip:
        parser.error(
            f"argument -ts: must be a multiple of {time_step_skip} {args.unit} (is {args.tau_step} {args.unit})")

    # Check 'tau_max'.
    if args.tau_max == -1:
        args.tau_max = (args.end - args.begin) / 2
        args.tau_max -= args.tau_max % args.tau_step
    if args.tau_max > args.end:
        parser.error(
            f"argument -tm: must be <= {args.end} {args.unit} (is {args.tau_max} {args.unit})")
    elif args.tau_max % args.tau_step or not args.tau_max // args.tau_step:
        parser.error(
            f"argument -tm: must be a multiple of {args.tau_step} {args.unit} (is {args.tau_max} {args.unit})")

    # Check 'PAF_times'.
    args.use_moving_PAF = False
    if len(args.PAF_times) > 1:
        args.PAF_times.pop(0)
    if -2 in args.PAF_times:
        args.use_moving_PAF = True
        args.PAF_times.remove(-2)
    if -1 in args.PAF_times:
        args.PAF_times.append(args.tau_step)
        args.PAF_times.remove(-1)
    for pt in args.PAF_times:
        if pt % args.tau_step:
            parser.error(
                f"argument -pt: must be -2, -1, or a multiple of {args.tau_step} {args.unit} (is {pt} {args.unit})")

    # Check 'tau_max_4fitting' and 'start_stop_step'.
    if args.SSS:
        args.tau_max_4fitting = np.arange(args.SSS[0],
                                          args.SSS[1] + args.SSS[2],
                                          args.SSS[2])
    for tmf in args.tau_max_4fitting:
        if tmf <= 0:
            parser.error(
                f"argument -tmf/-sss: must be > 0 (is {tmf} {args.unit})")
        elif tmf > args.tau_max:
            parser.error(
                f"argument -tmf/-sss: must be <= {args.tau_max} {args.unit} (is {tmf} {args.unit})")
        elif tmf % args.tau_step:
            parser.error(
                f"argument -tmf/-sss: must be multiple of {args.tau_step} {args.unit} (is {tmf} {args.unit})")

    # Check 'n_chunks'.
    if args.n_chunks < 0:
        parser.error(
            f"argument -nc: must be non-negative (is {args.n_chunks})")
    elif args.n_chunks > 0 and len(args.files) % args.n_chunks:
        parser.error(
            f"argument -nc: must be 0 or a divisor of {len(args.files)} (is {args.n_chunks})")
    elif args.n_chunks > 0 and len(args.files) // 2 < args.n_chunks:
        parser.error(
            f"argument -nc: must be <= {len(args.files) // 2} (is {args.n_chunks})")
    elif args.n_chunks != 0 and args.n_chunks < 2:
        parser.error(f"argument -nc: must be >= 2 (is {args.n_chunks})")

    # Set shadow parameters.
    args.time_step = time_step_skip
    args.ndx_start = int(args.begin / time_step_data)
    args.ndx_stop = int(args.end / time_step_data) + 1
    return parser, args
