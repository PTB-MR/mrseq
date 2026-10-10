import os
import time

import numpy as np
import pypulseq as pp
import yaml

# This file generates a sequence for the Duyn method. It also incorporates the MFC, so that one can acquire the same
# measurement from both the MFC and the Scanner (provided the MFC is set up)

# Main folder to keep everything in
main_dir = 'TriangleExample/'

# Main YAML
yaml_dir = 'TriangleExample/triangle_parameters.yaml'

# Add a custom name to the file to differentiate between measurements
add = 'Versuch_2'

# Every measurement with a new name gets saved as in a folder with a date stamp
output_path = main_dir + f'/{time.strftime("%Y%m%d")}_' + str(add)
print('output path: ', output_path)

# Check and make if directory doesn't exist
if os.path.exists(output_path) == False:
    os.mkdir(f'{output_path}')


# Get Parameters from YAML
with open(yaml_dir) as stream:
    try:
        # print(yaml.safe_load(stream))
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# Step 1 - Define our sequence parameters according to our scanner
# Create PyPulseq Sequence Object and Set System Limits
seq = pp.Sequence()
system = pp.Opts(
    max_grad=parameters['max_grad'] * 1e3,
    grad_unit='mT/m',
    max_slew=parameters['slew_rate'],
    slew_unit='T/m/s',
    rf_ringdown_time=parameters['rf_ringdown_time'],
    rf_dead_time=parameters['rf_dead_time'],
    adc_dead_time=parameters['adc_dead_time'],
)

method = ['Duyn']

# Quick Dynamics Calculation (necessary for the Field Cam but also helpful to see how many measurement instances)
CamNrDynamics = (
    len(parameters['enumerate_coeff'])
    * len(parameters['SLICE_POS'])
    * len(parameters['GRAD'])
    * len(parameters['RISE_TIMES'])
    * parameters['N_AVG']
)
print(f'CameraNrDynamics: {CamNrDynamics}')


# Create and Set Sinc Pulse Parameters
rf, g_sl, g_sl_r = pp.make_sinc_pulse(
    flip_angle=parameters['RF_PHI'] / 180 * np.pi,
    delay=system.rf_dead_time,
    duration=parameters['RF_DUR'],
    slice_thickness=parameters['SLICE_THICKNESS'],
    apodization=parameters['apodization'],
    time_bw_product=parameters['RF_BWT_PROD'],
    system=system,
    return_gz=True,
)


# Necessary delays such as TR, EddyCurrentComp, Trig for MFC
grad_free_time = pp.make_delay(parameters['grad_free'])  # from skope_gtf.m
delay_tr = pp.make_delay(parameters['TR'])  # TR delay
delay_ec = pp.make_delay(parameters['ec_delay'])  # eddy current compensation delay


# Step 6 - Actually making the measurement method take in data, so trig for MFC and ADC for Scanner (our beloved)
trig = pp.make_digital_output_pulse(
    channel='ext1', duration=parameters['trig_duration'], delay=0
)  # creating the MFC trigger pulse

# Automatic ADC samples and duration
adc_num_samples = int(np.round(parameters['adc_duration'] / parameters['dwell_time']))
print('adc_num: ', adc_num_samples)

# ADC Block
adc = pp.make_adc(
    num_samples=adc_num_samples, duration=parameters['adc_duration'], system=system, delay=system.adc_dead_time
)  # Analog Digital Converter


# Step 7 - Creating the loop

# Starting with averages
for avg in range(parameters['N_AVG']):
    avg_str = 'N_AVG/'  # Labels for loop order
    avg_label = pp.make_label(type='SET', label='AVG', value=avg)

    # We enumerate over rise times to increase the amplitude of the encode gradient
    for idx_rise_time, rise_time in enumerate(parameters['RISE_TIMES']):
        rise_times_str = 'RiseTimes/'
        seg_label = pp.make_label(type='SET', label='SEG', value=idx_rise_time)
        rise_time = np.round(
            rise_time, decimals=6
        )  # Not rounding the rise time here causes an error during amplitude calculation

        # Create ADC labels with label SEG
        for idx_grad, grad in enumerate(parameters['GRAD']):
            grad_str = 'Grad/'
            grad_label = pp.make_label(type='SET', label='SET', value=idx_grad)

            # Make labels of each axis with the index of the gradients
            for idx_slice_pos, slice_pos in enumerate(parameters['SLICE_POS']):
                slice_pos_str = 'SlicePos/'
                # Changing SLICE_POS to differentiate between Duyn and Rahmer (this is irrelevant for Duyn method)
                slice_label = pp.make_label(type='SET', label='SLC', value=idx_slice_pos)

                # amp_fac goes between -1 and 0 for Duyn
                for idx_amp_fac, amp_fac in enumerate(parameters['enumerate_coeff']):
                    enco_str = 'EnCo/'
                    amp_fac_label = pp.make_label(type='SET', label='REP', value=idx_amp_fac)

                    amp = amp_fac * system.max_slew * (rise_time)
                    # Starting amp factor * how fast amp rises * how long it rises for = amplitude value at the end of rise

                    # Add RF Pulse with Slice Selection
                    rf.freq_offset = g_sl.amplitude * slice_pos
                    # grad_slice_select amplitude * slice pos =>
                    # Gr(t)*Dr = freq difference of the rf pulse
                    g_sl.channel = grad
                    g_sl_r.channel = grad
                    seq.add_block(rf, g_sl)  # RF Pulse and the Slice Select Grad together
                    seq.add_block(g_sl_r)  # Slice Select Rewinder

                    # List of labels for everything
                    label_contents = [
                        avg_label,
                        seg_label,
                        grad_label,
                        slice_label,
                        amp_fac_label,
                    ]  # All labels, average, segment, gradient, slice and amp_factor

                    # Add Eddy Current Compensation Delay

                    seq.add_block(delay_ec)

                    # Add ADC and Triangle Gradient
                    if amp_fac != 0:  # If there is an amp (if gradient present)
                        # Create Triangle Gradient

                        g_triangle = pp.make_trapezoid(
                            channel=grad, system=system, rise_time=rise_time, flat_time=0, amplitude=amp, delay=0
                        )
                        # trapezoid with no flat time = triangle
                        # enumerating over rise times gives differently sized triangles
                        # Watch out for delay in the recon

                        seq.add_block(trig)  # MFC trigger
                        # starts CameraAcqDuration, not relevant if no MFC present

                        seq.add_block(grad_free_time)  # Delay between MFC trig and ADC

                        seq.add_block(adc, g_triangle)  # ADC and gradients during the same block

                    else:
                        seq.add_block(trig)  # MFC trigger

                        seq.add_block(grad_free_time)  # Delay between MFC trig and ADC
                        seq.add_block(adc)  # ADC without the gradients

                    # Add TR Delay
                    seq.add_block(delay_tr)

    seq.add_block(avg_label)

# Total loop order string for recon
label_str = avg_str + rise_times_str + grad_str + slice_pos_str + enco_str

# Step 8 - Timing Check (Also good for sanity)
ok, error_report = seq.check_timing()
if ok:
    print('\nTiming check passed successfully')
else:
    print('\nTiming check failed! Error listing follows\n')
    print(error_report)


# Step 9 - Setting Definitions for the seq file (to be used by the scanner and the MFC)
seq.set_definition('rise_time', parameters['RISE_TIMES'])
seq.set_definition('grad', parameters['GRAD'])
seq.set_definition('slice_pos', parameters['SLICE_POS'])
seq.set_definition('avg', parameters['N_AVG'])
seq.set_definition('tr_delay', parameters['TR'])
seq.set_definition('Acquisition Method: ', method)
#
seq.set_definition('CameraNrDynamics', CamNrDynamics)
# CameraNrDynamics: Maximum number of acquisitions performed by the Camera Acquisition System.
seq.set_definition('CameraNrSyncDynamics', parameters['cam_nr_sync_dyn'])
seq.set_definition('CameraAcqDuration', parameters['cam_acq_duration'])
seq.set_definition('CameraInterleaveTR', parameters['cam_interleave_tr'])
seq.set_definition('CameraAqDelay', parameters['cam_acq_delay'])
# I find it helpful to print out some important parameters
# Especially ones you need to differentiate between measurements

# Step 10 - Save File (and maybe even plot)
filename = f'{output_path}/{time.strftime("%Y%m%d")}_Triangles'


print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
seq.write(str(filename), create_signature=True)

# seq.plot(label="avg", save=True)
seq.plot(save=False, grad_disp='mT/m')  # Plot the sequence and display the gradients in mT/m

# Write the loop order into the YAML file to be stored
with open(f'{output_path}/{add}.yaml', 'w+') as f:
    yaml.dump(parameters, f, sort_keys=False)

# In the end, you will have specifically named folders containing a sequence and the YAML parameters used
# to create that exact sequence, so that there is no confusion during reconstruction
