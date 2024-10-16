def Interpolate_combine(data1, data2, name1, name2) :
    # 데이터 보간.
    import os
    import pandas as pd

    # 파일 읽기 (CSV 파일이므로 read_csv 사용)
    df1 = data1
    df2 = data2

    # 보간할 가중치 설정
    weight_1 = 0.8  # 파일1에 더 중점을 둠
    weight_2 = 0.2  # 파일2에 덜 중점을 둠

    output = 'dataset/Interpolated_FTIR/'

    # 가중치가 합이 1이 되도록 설정
    weight_1 /= (weight_1 + weight_2)
    weight_2 /= (weight_1 + weight_2)

    # 보간 함수 정의
    def weighted_linear_interpolation(df1, df2, w1, w2):
        # 두 데이터프레임 중 더 작은 길이에 맞춰 자르기
        minimize = min(len(df1), len(df2))

        # 숫자가 아닌 값이 있을 경우 이를 NaN으로 변환 (pd.to_numeric 사용)
        df1_numeric = pd.to_numeric(df1.iloc[:minimize], errors='coerce')
        df2_numeric = pd.to_numeric(df2.iloc[:minimize], errors='coerce')

        # NaN 값을 처리 (필요시 dropna 또는 fillna 사용)
        df1_numeric = df1_numeric.dropna()
        df2_numeric = df2_numeric.dropna()

        # 가중치 적용한 선형 보간 (벡터 연산 사용)
        result = df1_numeric * w1 + df2_numeric * w2

        return result

    # 각 열에 대해 보간 적용
    df_interpolated = pd.DataFrame({
        'cm': weighted_linear_interpolation(df1[0], df2[0], weight_1, weight_2),
        '%': weighted_linear_interpolation(df1[1], df2[1], weight_1, weight_2),
    })

    # 가중치를 문자열로 변환하여 파일명에 포함
    output_file_name = f"{name1}_{name2}_w{weight_1:.2f}_w{weight_2:.2f}_aug_.csv"
    output_dir = output

    # 파일 경로에 숫자를 증가시키는 로직
    i = 1
    while os.path.exists(os.path.join(output_dir, output_file_name)):
        output_file_name = f"{name1}_{name2}_w{weight_1:.2f}_w{weight_2:.2f}_{i}.csv"
        i += 1

    # 최종 저장 경로
    output_file_path = os.path.join(output_dir, output_file_name)

    # 새로운 CSV 파일로 저장
    df_interpolated.to_csv(output_file_path, index=False)

    print(f"보간된 데이터가 저장되었습니다: {output_file_path}")

