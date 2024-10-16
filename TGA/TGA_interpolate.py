import numpy as np

from utill import draw_graph
from TGA import TGA_to_csv


def interpolate(data_for_return, temp1, temp2) :
    for cat in ['No', 'PtC', 'RuC', 'RN']:
        for i in range(1, 151):

            if i % 50 == 0:
                continue

            target_temp = 250 + i

            start_temp = 250 + (50 * (target_temp // 50) - 5)
            limit_temp = 300 + (50 * (target_temp // 50) - 5)

            percent = calculate.calculate_percent(start_temp, limit_temp, target_temp)


            minimize = min(len(data_for_return[temp1][2]), len(data_for_return[temp2][2]))


            predict_temperature = calculate.linear_interpolation(data_for_return[temp1][6][:minimize], data_for_return[temp2][6][:minimize], percent * 0.01)
            predict_percent = calculate.linear_interpolation(data_for_return[temp1][3][:minimize], data_for_return[temp2][3][:minimize], percent * 0.01)
            predict_data = calculate.linear_interpolation(data_for_return[temp1][4][:minimize], data_for_return[temp2][4][:minimize], percent * 0.01)

            minimize -= 1

            show_predict_graph = False
            if show_predict_graph :
                label2 = "predict " + str(target_temp)
                draw_graph.draw_gp(np.abs(data_for_return[temp1][4][minimize]), predict_data[:minimize], np.abs(data_for_return[temp2][4][:minimize]), data_for_return[temp2][6][:minimize], str(start_temp), label2, str(limit_temp), "300")

            file_name = f"predict_{cat}_{target_temp}"

            # 시간, 온도, 무게%, 변화량
            TGA_to_csv.save_custom_format_csv(data_for_return[temp1][6][:minimize], predict_temperature, predict_percent, predict_data, file_name)
