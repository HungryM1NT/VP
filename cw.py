import numpy as np
import pandas as pd


def generate_request_time_linear(ts_min, ts_max, work_time):
    T_server = [0]
    while True:
        new_t = T_server[-1] + (ts_max - ts_min) * np.random.rand() + ts_min
        if new_t > work_time:
            break
        T_server.append(new_t)
    T_server.pop(0)
    return np.asarray(T_server)

def generate_process_time_linear(tz_min, tz_max, n):
    return (tz_max - tz_min) * np.random.rand(n) + tz_min

def exp_func(lmbd, p):
    return -1 / lmbd * np.log(p)

def generate_request_time_exp(lmbd, work_time):
    T_server = [0]
    while True:
        new_t = T_server[-1] + exp_func(lmbd, np.random.rand())
        if new_t > work_time:
            break
        T_server.append(new_t)
    T_server.pop(0)
    return np.asarray(T_server)

def generate_process_time_exp(lmbd, n):
    return exp_func(lmbd, np.random.rand(n))

def get_f_o_t(cur_t, last_start_s1, last_end_s1, last_start_s2, last_end_s2):
    last_t = max(last_start_s1, last_start_s2)
    two = max(0, (min(last_end_s1, last_end_s2) - last_t))
    free = max(0, cur_t - max(last_end_s1, last_end_s2))
    one = cur_t - last_t - two - free
    return (free, one, two)

def simulate(Ts, Tz, work_time):
    last_start_s1 = 0
    last_end_s1 = 0
    last_start_s2 = 0
    last_end_s2 = 0
    processed_signals = 0

    free_time = 0
    one_work_time = 0
    two_work_time = 0

    for i in range(len(Ts)):
        if Ts[i] + Tz[i] <= work_time:
            if last_end_s1 <= Ts[i]:
                free, one, two = get_f_o_t(Ts[i], last_start_s1, last_end_s1, last_start_s2, last_end_s2)
                free_time += free
                one_work_time += one
                two_work_time += two

                last_start_s1, last_end_s1 = Ts[i], Ts[i] + Tz[i]
                processed_signals += 1
            elif last_end_s2 <= Ts[i]:
                free, one, two = get_f_o_t(Ts[i], last_start_s1, last_end_s1, last_start_s2, last_end_s2)
                free_time += free
                one_work_time += one
                two_work_time += two

                last_start_s2, last_end_s2 = Ts[i], Ts[i] + Tz[i]
                processed_signals += 1
    
    free, one, two = get_f_o_t(work_time, last_start_s1, last_end_s1, last_start_s2, last_end_s2)
    free_time += free
    one_work_time += one
    two_work_time += two

    return (processed_signals, (free_time, one_work_time, two_work_time))


work_time = 1 * 60 * 60

# Linear
tz_min = 1 / 2
tz_max = 5 / 6

ts_min = 1
ts_max = 5

# Exp
lmbd = 1.5
t_proc = 2

mu = 1 / t_proc

# Exp table
table_1_vals = np.zeros((5, 6))

for i in range(5):
    Ts = generate_request_time_exp(lmbd, work_time)
    Tz = generate_process_time_exp(mu, len(Ts))

    processed_requests, fot_times = simulate(Ts, Tz, work_time)

    p0 = fot_times[0] / work_time
    p1 = fot_times[1] / work_time
    p2 = fot_times[2] / work_time

    Q = processed_requests / len(Ts)
    A = processed_requests / work_time
    k = (fot_times[1] * 1 + fot_times[2] * 2) / work_time

    table_1_vals[i, :] = [p0, p1, p2, Q, A, k]

table_1 = pd.DataFrame(table_1_vals, columns=['P0', 'P1', 'P2', 'Q', 'A', 'k'], index=np.arange(1, 6))
print("Результаты работы программы при пяти запусках")
print(table_1)
print()

# Exp and linear table
table_2_vals = np.zeros((12, 6))

for i in range(10):
    Ts_exp = generate_request_time_exp(lmbd, work_time)
    Tz_exp = generate_process_time_exp(mu, len(Ts_exp))

    processed_requests_exp, fot_times_exp = simulate(Ts_exp, Tz_exp, work_time)

    p0_exp = fot_times_exp[0] / work_time
    p1_exp = fot_times_exp[1] / work_time
    p2_exp = fot_times_exp[2] / work_time
    
    Ts_lin = generate_request_time_linear(tz_min, tz_max, work_time)
    Tz_lin = generate_process_time_linear(ts_min, ts_max, len(Ts_lin))

    processed_requests_lin, fot_times_lin = simulate(Ts_lin, Tz_lin, work_time)
    p0_lin = fot_times_lin[0] / work_time
    p1_lin = fot_times_lin[1] / work_time
    p2_lin = fot_times_lin[2] / work_time

    table_2_vals[i, :] = [p0_exp, p1_exp, p2_exp, p0_lin, p1_lin, p2_lin]

table_2_vals[-2,] = np.mean(table_2_vals[:-2], axis=0)
table_2_vals[-1,] = np.std(table_2_vals[:-2], axis=0)

table_2_rows = list(range(1, 11))
table_2_rows.extend(["M", "S"])
table_2 = pd.DataFrame(np.round(table_2_vals, 6), columns=['P0', 'P1', 'P2', 'PO', 'P1', 'P2'], index=table_2_rows)
print("Сопоставление результатов для линейного распределения")
print("Результаты тестовых расчетов=======Результаты рабочих расчетов")
print(table_2)
print()

# Task 5
table_3_vals = np.zeros((2, 6))

all_params_exp = np.zeros((10, 6))
all_params_lin = np.zeros((10, 6))
for i in range(10):
    Ts_exp = generate_request_time_exp(lmbd, work_time)
    Tz_exp = generate_process_time_exp(mu, len(Ts_exp))

    processed_requests_exp, fot_times_exp = simulate(Ts_exp, Tz_exp, work_time)

    p0_exp = fot_times_exp[0] / work_time
    p1_exp = fot_times_exp[1] / work_time
    p2_exp = fot_times_exp[2] / work_time
    Q_exp = processed_requests_exp / len(Ts_exp)
    A_exp = processed_requests_exp / work_time
    k_exp = (fot_times_exp[1] * 1 + fot_times_exp[2] * 2) / work_time

    all_params_exp[i] = [p0_exp, p1_exp, p2_exp, Q_exp, A_exp, k_exp]

    Ts_lin = generate_request_time_linear(tz_min, tz_max, work_time)
    Tz_lin = generate_process_time_linear(ts_min, ts_max, len(Ts_lin))

    processed_requests_lin, fot_times_lin = simulate(Ts_lin, Tz_lin, work_time)
    p0_lin = fot_times_lin[0] / work_time
    p1_lin = fot_times_lin[1] / work_time
    p2_lin = fot_times_lin[2] / work_time
    Q_lin = processed_requests_lin / len(Ts_lin)
    A_lin = processed_requests_lin / work_time
    k_lin = (fot_times_lin[1] * 1 + fot_times_lin[2] * 2) / work_time
    
    all_params_lin[i] = [p0_lin, p1_lin, p2_lin, Q_lin, A_lin, k_lin]

table_3_vals[0, :] = np.mean(all_params_exp, axis=0)
table_3_vals[1, :] = np.mean(all_params_lin, axis=0)
df = pd.DataFrame(table_3_vals, columns=['P0', 'P1', 'P2', 'Q', 'A', 'k'], index=["Exp", "Lin"])
print("Выходные характеристики ВС")
print(df)