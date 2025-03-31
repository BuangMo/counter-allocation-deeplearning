from math import sqrt

def main():
    debug = True

    # at time instance 4s, the chosen countermeasures cover pulses for both channels
    # we know from the data that both threats 10 and 8 are still in the search radar stage
    se_id1 = 0.8            # stage effectivenes of channel 1
    se_id2 = 0.9            # stage effectivenes of channel 1

    # next we look at the effect of the techniques on each other
    i_id1 = 1.0             # the effect of channel 2 on channel 1 (no interference)
    i_id2 = 1.0             # the effect of channel 1 on channel 2 (no interference)

    # next we go to the cross-effect
    ce_id1 = 0.8            # the countermesure is optimised for threat type 10 and we are examining threat type 10
    ce_id2 = 0.8            # the countermesure is optimised for threat type 8 and we are examining threat type 8

    # next we go for the chaff interference (CI) and since no chaff is used, this value is set to 1
    ci_id1 = 1.0            # the interference of the chaff technique on the technique of active channel 1
    ci_id2 = 1.0            # the interference of the chaff technique on the technique of active channel 2

    # the antenna gains for the examined threats at the time instance are
    ag2_id1 = 0.0227
    ag2_id2 = 0.7760

    # lastly, we go onto the countermeasure resistance factors, for this we have to get the countemeasure resistance of 
    # the threats
    cr_id2 = 0.1          # the countermeasure resistance of threat type 10
    cr_id1 = 0.2           # the countermeasure resistance of threat type 8

    # thus the countermeasure resistance factors of the two threats selected are
    # threat type 10
    if se_id2 >= 0:
        cr_factor_id2 = 1 - cr_id2
    else:
        cr_factor_id2 = 1 + cr_id2

    # threat type 8
    if se_id1 >= 0:
        cr_factor_id1 = 1 - cr_id1
    else:
        cr_factor_id1 = 1 + cr_id1

    # now, the stage effect of active channel 1 and 2 are thus
    jeffect_id1 = ag2_id1 * se_id1 * i_id1 * ce_id1 * ci_id1 * cr_factor_id1
    jeffect_id2 = ag2_id2 * se_id2 * i_id2 * ce_id2 * ci_id2 * cr_factor_id2

    if debug:
        print(f'The jamming effect of active channel 1 is {jeffect_id1}')
        print(f'The jamming effect of active channel 2 is {jeffect_id2}\n')

    ######### total jamming effect
    # now the coordinates of the platform at time instance 4
    pl_coord = [10, 11.43, 8]

    # the threats co-ordinates are
    id1_coord = [12, 8, 0]
    id2_coord = [7, 14, 0]

    # calculation of slant distances
    id2_da = sqrt((id2_coord[0] - pl_coord[0])**2 + (id2_coord[1] - pl_coord[1])**2 + (id2_coord[2] - pl_coord[2])**2)
    id1_da = sqrt((id1_coord[0] - pl_coord[0])**2 + (id1_coord[1] - pl_coord[1])**2 + (id1_coord[2] - pl_coord[2])**2)

    # combined jamming effect of active channel for threat type 8 and 10
    je_adj_id1 = jeffect_id1 * (1 + (id1_da / 20)**2)
    je_adj_id2 = jeffect_id2 * (1 + (id2_da / 20)**2)

    if debug:
        print(f'The adjusted jamming effect for threat ID 1 is {je_adj_id1}')
        print(f'The adjusted jamming effect for threat ID 2 is {je_adj_id2}\n')

if __name__ == '__main__':
    main()