def process_group_for_TGA(data, cat, temperature, test=False):
    '''
    Args:
        data: TGA data
        cat:  catalyst
        temperature: data temparature
        test:

    Returns:

    '''
    # 촉매 그룹 설정: NoCat, Pt/C, Ru/C, Raney Ni
    group_number = {
        'NoCat': [0, 4, 8, 12],
        'PtC': [1, 5, 9, 13],
        'RuC': [2, 6, 10, 14],
        'RN': [3, 7, 11, 15]
    }

    if cat not in group_number:
        raise ValueError("허용되지 않은 촉매")

    if not test:
        if 250 < temperature < 300:
            tem1, tem2 = 0, 1
        elif 300 < temperature < 350:
            tem1, tem2 = 1, 2
        elif 350 < temperature < 400:
            tem1, tem2 = 2, 3
        else:
            raise ValueError("허용되지 않은 온도")
    else:
        tem1, tem2 = 0, 2

    # 선택한 그룹의 모든 데이터를 반환
    selected_group = group_number[cat]
    data_for_return = [data[i] for i in selected_group]

    return data_for_return, tem1, tem2

def process_group_for_FTIR(data, cat, temperature, test = False) :
    # No Cat, Pt/C, Ru/C, Raney Ni
    group_number = [[0, 12, 24, 36], [3, 15, 27, 39], [6, 18, 30, 42], [9, 21, 33, 45]]
    if cat == 'No':
        choice = 0
    elif cat == 'Pt/C':
        choice = 1
    elif cat == 'Ru/C':
        choice = 2
    elif cat == 'Raney Ni':
        choice = 3
    else:
        raise ValueError("허용되지 않은 촉매")

    if not(test) :
        if temperature == 250 :
            tem1, tem2 = 0, 0
        elif temperature == 300:
            tem1, tem2 = 1, 1
        elif temperature == 350:
            tem1, tem2 = 2, 2
        elif temperature == 400:
            tem1, tem2 = 3, 3

        elif 250 < temperature < 300 :
            tem1, tem2 = 0, 1
        elif 300 < temperature < 350 :
            tem1, tem2 = 1, 2
        elif 350 < temperature < 400 :
            tem1, tem2 = 2, 3
        else :
            raise ValueError("허용되지 않은 온도")

    else :
        tem1, tem2 = 0, 2

    data_for_return = []

    for tem in [tem1, tem2] :

        add = []

        for i in range(3):
            add.append(data[group_number[choice][tem] + i][1])

        data_for_return.append(add)

    return data_for_return, tem1, tem2