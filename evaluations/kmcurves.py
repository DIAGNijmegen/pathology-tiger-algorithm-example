import statistics

import altair as alt
import pandas as pd
from lifelines import KaplanMeierFitter

INPUT_SAV_FILE = "/home/user/results.sav"
OUTPUT_JSON_FILE = "/output/kaplan_meier_curve.json"

def main():
    df = pd.read_spss(INPUT_SAV_FILE)

    E = df.recurrence
    T = df.time_recurrence
    kmf1 = KaplanMeierFitter()  
    kmf2 = KaplanMeierFitter()  

    groups = df.TILscore
    i1 = groups < statistics.median(
        df.TILscore
    )  
    i2 = groups >= statistics.median(
        df.TILscore
    )  

    if all(i1):
        i2 = i1
    if all(i2):
        i1 = i2

    ## fit the model for 1st cohort
    kmf1.fit(T[i1], E[i1], label="Under median")
    a1 = kmf1.plot()

    ## fit the model for 2nd cohort
    kmf2.fit(T[i2], E[i2], label="Above median")
    kmf2.plot(ax=a1)


    df_plot = kmf1.survival_function_.copy(deep=True)
    df_plot["lower_bound"] = kmf1.confidence_interval_["Under median_lower_0.95"]
    df_plot["upper_bound"] = kmf1.confidence_interval_["Under median_upper_0.95"]
    df_plot.reset_index(inplace=True)

    df_plot2 = kmf2.survival_function_.copy(deep=True)
    df_plot2["lower_bound"] = kmf2.confidence_interval_["Above median_lower_0.95"]
    df_plot2["upper_bound"] = kmf2.confidence_interval_["Above median_upper_0.95"]
    df_plot2.reset_index(inplace=True)


    line = (
        alt.Chart(df_plot)
        .mark_line(interpolate='step-after', color='blue')
        .transform_fold(
            fold=['Above median'])
        .encode(
            x=alt.X("timeline", axis=alt.Axis(title="Months")),
            y=alt.Y("Under median", axis=alt.Axis(title="Survival probability"), scale=alt.Scale(domain=[min(df_plot['Under median'])-0.1, 1.0]))
        )
    )

    band = line.mark_area(opacity=0.4, color='blue').encode(
        x='timeline',
        y='lower_bound',
        y2='upper_bound'
    )

    line2 = (
        alt.Chart(df_plot2)
        .mark_line(interpolate='step-after', color='orange', )
        ).encode(
            x=alt.X("timeline", axis=alt.Axis(title="Months")),
            y=alt.Y("Above median",  scale=alt.Scale(domain=[min(df_plot2['Above median'])-0.1, 1.0])),
        )


    band2 = line2.mark_area(opacity=0.4, interpolate='step-after', color='orange').encode(
        x='timeline',
        y='lower_bound',
        y2='upper_bound'
    )

    text1 = alt.Chart({'values':[{}]}).mark_text(
        align="left", baseline="top", color='orange',
    ).encode(
        x=alt.value(250),  # pixels from left
        y=alt.value(5),  # pixels from top
        text=alt.value(['--- Above median']))

    text2 = alt.Chart({'values':[{}]}).mark_text(
        align="left", baseline="top", color='blue'
    ).encode(
        x=alt.value(250),  # pixels from left
        y=alt.value(15),  # pixels from top
        text=alt.value(["--- Under median"]))

    box = alt.Chart({'values':[{}]}).mark_rect(opacity=0.2, stroke='black', color='white').encode(
        x=alt.value(245),
        x2=alt.value(398),
        y=alt.value(4),
        y2=alt.value(30))


    # fig = line + band + line2 + band2 + box + text1 +text2
    fig = line + line2 + box + text1 +text2
    print(f'Saving file {OUTPUT_JSON_FILE}')
    fig.save(OUTPUT_JSON_FILE)


if __name__ == "__main__":
    main()