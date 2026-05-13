import pandas as pd
import numpy as np

# 加载数据
def load_data(file_path):
    return pd.read_excel(file_path)

# 提取特定列的数据并计算统计值
def calculate_statistics(data, column_name):
    values = data[column_name].dropna().values  # 提取目标列的非空值
    best_value = np.min(values)  # 最佳值（最小值）
    worst_value = np.max(values)  # 最差值（最大值）
    median_value = np.median(values)  # 中位数
    mean_value = np.mean(values)  # 平均值
    std_dev = np.std(values)  # 标准差
    return {
        "Best": best_value,
        "Worst": worst_value,
        "Median": median_value,
        "Mean": mean_value,
        "StdDev": std_dev
    }

# 比较两个算法的性能
def compare_performance(algo1_stats, algo2_stats, metric="Mean", threshold=0.05):
    """
    比较两个算法的性能。
    :param algo1_stats: 算法1的统计值
    :param algo2_stats: 算法2的统计值
    :param metric: 比较的指标(默认为"Mean")
    :param threshold: 差异阈值，小于该值认为性能相当
    :return: 符号表示性能比较结果 ("+", "-", "=")
    """
    value1 = algo1_stats[metric]
    value2 = algo2_stats[metric]

    if abs(value1 - value2) < threshold:
        return "="  # 性能相当
    elif value1 < value2:
        return "+"  # 算法1优于算法2
    else:
        return "-"  # 算法1劣于算法2

if __name__ == "__main__":
    file_paths = {
        "AHDPSO":r"D:\桌面\M_PSO\相同时间\不固定\AHDPSO.xlsx",
        "AFMPSO":r"D:\桌面\M_PSO\相同时间\不固定\AFMPSO.xlsx",
        "AMSEPSO":r"D:\桌面\M_PSO\相同时间\不固定\AMSEPSO.xlsx",
        "FHPSO":r"D:\桌面\M_PSO\相同时间\不固定\FHPSO.xlsx",
        "SAOMPSO":r"D:\桌面\M_PSO\相同时间\不固定\SAOMPSO.xlsx",
        "SDPSO": r"D:\桌面\M_PSO\相同时间\不固定\SDPSO.xlsx",
        "SPSO": r"D:\桌面\M_PSO\相同时间\不固定\SPSO.xlsx",
        "MPSO_2020":r"D:\桌面\M_PSO\相同时间\不固定\MPSO2020.xlsx",
        "MPSO_2021":r"D:\桌面\M_PSO\相同时间\不固定\MPSO2021.xlsx",
        "PSO":r"D:\桌面\M_PSO\相同时间\不固定\PSO.xlsx"
    }
    
    # 加载数据
    datasets = {name: load_data(path) for name, path in file_paths.items()}

    target_column = "Fitness"

    # 计算每个算法的统计值
    statistics = {}
    for algo_name, dataset in datasets.items():
        stats = calculate_statistics(dataset, target_column)
        statistics[algo_name] = stats

    # 输出统计结果
    print("Algorithm Statistics:")
    headers = ["Algorithm", "Best", "Worst", "Median", "Mean", "StdDev"]
    row_format = "{:<15}" * len(headers)
    print(row_format.format(*headers))
    print("-" * 80)

    for algo_name, stats in statistics.items():
        row = [algo_name] + [f"{stats[key]:.6f}" for key in headers[1:]]
        print(row_format.format(*row))

    # 比较 AHDPSO 与其他算法的性能
    print("\nPerformance Comparison with AHDPSO:")
    comparison_metric = "Mean"  # 使用平均值作为比较指标
    threshold = 0.001  # 性能差异阈值

    afmpso_stats = statistics["AHDPSO"]
    comparison_results = {}

    for algo_name, algo_stats in statistics.items():
        if algo_name != "AHDPSO":
            symbol = compare_performance(afmpso_stats, algo_stats, metric=comparison_metric, threshold=threshold)
            comparison_results[algo_name] = symbol

    # 输出比较结果
    header = ["Algorithm", "Comparison Symbol"]
    row_format = "{:<15}{:<15}"
    print(row_format.format(*header))
    print("-" * 30)

    for algo_name, symbol in comparison_results.items():
        print(row_format.format(algo_name, symbol))
