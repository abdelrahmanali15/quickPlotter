

from PyLTSpice import RawRead
from PlotManager import PlotManager

if __name__ == "__main__":

    style = 'prof'
    pm = PlotManager(style=style)
    LTR = RawRead("tests\simple_rc_tran.raw")
    # print(LTR.get_trace_names())
    # print(LTR.get_raw_property())

    v1 = LTR.get_trace("V(1)")
    x = LTR.get_trace('time')  # Gets the time axis
    pm.quick_plot(x=x.get_wave(),
                  y=v1.get_wave(),
                  )

    v2 = LTR.get_trace("V(2)")
    steps = LTR.get_steps()
    for step in range(len(steps)):
        # print(steps[step])
        # plt.plot(x.get_wave(step), v2.get_wave(step), label=steps[step])
        pm.quick_plot(x=x.get_wave(step),
                      y=v2.get_wave(step),
                      x_label="$Time (sec)$",
                      y_label="$Amplitude (V)$",
                      title="$Transient\;Analysis$",
                      legend=['$Vin$', '$Vout@CPAR=500pF$',
                              '$Vout@CPAR=1nF$', '$Vout@CPAR=1.5nF$'],
                      y_scale='linear'
                      )

    # pm.add_axis()
#     del pm
    pm2 = PlotManager(style=style)
    LTR2 = RawRead("tests/simple_rc_ac.raw")

    import numpy as np

    x = LTR2.get_trace('frequency')  # Gets the time axis

    v2 = LTR2.get_trace("V(2)")

    steps = LTR2.get_steps()
    # print(magV2)
    for step in range(len(steps)):
        # print(steps[step])
        # plt.plot(x.get_wave(step), v2.get_wave(step), label=steps[step])
        freq = np.abs(x.get_wave(step))
        magV2 = 20 * np.log10(np.abs(v2.get_wave(step)))
        phV2 = np.angle(v2.get_wave(step), deg=True)
        pm2.quick_plot(x=freq,
                       y=magV2,
                       x_label="$Frequency (Hz)$",
                       y_label="$Gain (dB)$",
                       title="$AC\;Analysis$",
                       legend=['$CPAR=500pF$',
                               '$CPAR=1nF$', '$CPAR=1.5nF$'],
                       y_scale='linear',
                       x_scale='log'
                       )
#     del pm2
    pm3 = PlotManager(style=style)
    for step in range(len(steps)):
        # print(steps[step])
        # plt.plot(x.get_wave(step), v2.get_wave(step), label=steps[step])
        freq = np.abs(x.get_wave(step))
        phV2 = np.angle(v2.get_wave(step), deg=True)
        pm3.quick_plot(x=freq,
                       y=phV2,
                       x_label="$Frequency (Hz)$",
                       y_label="$Phase (deg)$",
                       title="$AC\;Analysis$",
                       legend=['$CPAR=500pF$',
                               '$CPAR=1nF$', '$CPAR=1.5nF$'],
                       y_scale='linear',
                       x_scale='log'
                       )

#     del pm3
    pm4 = PlotManager(style=style)
    # LTR = RawRead("tests/opamp_tran_tb.raw")
    # print(LTR.get_trace_names())
    # print(LTR.get_raw_property())

    # out = LTR.get_trace("V(out)")
    # pos = LTR.get_trace("V(pos)")
    # x = LTR.get_trace('time')  # Gets the time axis

    # pm4.quick_plot(x=[x.get_wave(), x.get_wave()],
    #                y=[pos.get_wave(), out.get_wave()],
    #                x_label="$Time (sec)$",
    #                y_label="$Amplitude (V)$",
    #                title="$Transient\;Analysis$",
    #                legend=['$Vin$', '$Vout@CPAR=500pF$',
    #                        '$Vout@CPAR=1nF$', '$Vout@CPAR=1.5nF$'],
    #                y_scale='linear'
    #                )

    pm4.show_plot()
