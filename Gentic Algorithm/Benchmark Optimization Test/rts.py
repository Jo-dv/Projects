import copy
import math

import numpy as np

from evaluate import evaluate
from fitness import fitness


def find_similar(c, d):
    similarity = np.sum(np.power(c - d, 2))  # 유사한 individuals을 찾기 위해 euclidean distance 사용
    return np.sqrt(similarity)


def rts_select(population, mutant, w, op):  # restricted tournament selection
    new_population = copy.deepcopy(population)  # steady-state 형태이므로 기존 population을 그대로 복사
    popsize = np.shape(population)[0]
    dim = np.shape(population[0])[0]

    for i in range(popsize):
        distance = math.inf  # 거리 초기화
        position = 0  # 초기 탐색 위치
        mutant_eval = evaluate(fitness(mutant[i].reshape(-1, dim), op))  # mutant의 fitness를 평가
        idx = [np.random.choice(range(0, popsize)) for _ in
               range(w)]  # population 내 individuals의 index를 window size만큼 무작위 선택

        comparison_set = population[idx]  # 선택된 index를 가진 index에 해당하는 individuals를 mutant와 비교하기 위해 배열에 저장
        for j in range(w):  # window size만큼 반복
            similarity = find_similar(mutant[i], comparison_set[j])  # mutant와 경쟁자의 유사도 계산
            if similarity < distance:  # 계산한 유사도가 distance보다 작을 경우
                distance = similarity  # 거리를 계산한 유사도로 갱신
                competitor = comparison_set[j]  # 배열에서 다음 경쟁자 선택
                position = idx[j]  # 가까운 경쟁자의 원 population 내 index를 저장

        competitor_eval = evaluate(fitness(competitor.reshape(-1, dim), op))  # 최종 경쟁자의 fitness를 평가

        if mutant_eval > competitor_eval:  # mutant가 최종 경쟁자의 평가보다 좋을 경우
            new_population[position] = mutant[i]  # mutant를 원 population 내, 최종 경쟁자와 교체함

    return new_population