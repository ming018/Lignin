from scipy.ndimage import gaussian_filter1d

def gaussian_smooth_data(data, sigma=1.0):
    """가우시안 필터를 사용하여 데이터 스무딩."""
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    return smoothed_data

