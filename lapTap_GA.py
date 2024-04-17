import random
import pandas as pd


'''적응도 함수'''
def fitness(individual, target):
    # 개체의 가격을 계산
    price = int(CPU[individual[0]][3]) + int(GPU[individual[1]][3]) + int(MB[individual[2]][2]) + int(PO[individual[3]][3]) + int(RAM[individual[4]][3]) + int(SSD[individual[5]][6])
    # 예산과의 차이를 계산
    difference = abs(target - price)
    # 차이가 0에 가까울수록 적응도가 높아야 하므로, 차이의 역수를 반환
    # 차이가 0일 경우를 대비해 1을 더해 분모가 0이 되는 것을 방지
    return 1 / (difference + 1)
##-----> 


'''GA_추가_기능'''
# 개체 생성 함수
def create_individual(items):
    return [random.choice(item) for item in items]

# 개체군 초기화
def init_population(items, population_size):
    return [create_individual(items) for _ in range(population_size)]

# 적합도 계산
def evaluate_population(population, target):
    return [fitness(individual, target) for individual in population]

# 선택 (여기서는 단순한 토너먼트 선택 사용)
def select(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=2)
    return max(selected, key=lambda ind: fitness(ind, target_price))

# 교차
def crossover(parent1, parent2):
    index = random.randint(1, len(parent1) - 2)
    return parent1[:index] + parent2[index:], parent2[:index] + parent1[index:]

# 변이
def mutate(individual):
    index = random.randint(0, len(individual) - 1)
    mutation = random.choice(items[index])
    individual[index] = mutation



# 부품별 가격 범위 설정
df_CPU = pd.read_csv("./data/CPU_LIST.csv", decimal=',')
df_GPU = pd.read_csv("./data/GPU_LIST.csv", decimal=',')
df_MB = pd.read_csv("./data/MAINBOARD_LIST.csv", decimal=',')
df_PO = pd.read_csv("./data/POWER_LIST.csv", decimal=',')
df_RAM = pd.read_csv("./data/RAM_CSV.csv", decimal=',')
df_SSD = pd.read_csv("./data/SSD_LIST.csv", decimal=',')


# 순위, 모델명, 점수, 가격
CPU = df_CPU.values
# 순위, 모델명, 점수, 가격
GPU = df_GPU.values
# 순위, 모델명, 가격
MB = df_MB.values
# 등급, 모델명, 와트, 가격
PO = df_PO.values
# 순위, 모델명, 성능, 가격
RAM = df_RAM.values
# 순위, 모델명, 유저 점수, 가성비, 벤치점수, 성능, 가격
SSD = df_SSD.values


cpu_list = { i for i in range(len(CPU))}  
gpu_list = { i for i in range(len(GPU))}  
mb_list = { i for i in range(len(MB))}  
po_list = { i for i in range(len(PO))}  
ram_list = { i for i in range(len(RAM))}  
ssd_list = { i for i in range(len(SSD))}


# 부품별 mapping을 위해 index로 조합
items = [
    list(cpu_list),
    list(gpu_list),
    list(mb_list),
    list(po_list),
    list(ram_list),
    list(ssd_list)    
]


# 유전 알고리즘 실행
def run_ga(items, target_price, population_size=100, generations=50):
    population = init_population(items, population_size)

    ##for i in range(len(population)):
      ##  print(f'{i+1}번째 sample = {population[i]}')

    for generation in range(generations):
        fitnesses = evaluate_population(population, target_price)

        ##if generation == 0 :
            ##for i in range(len(fitnesses)):
               ## print(f'{i}번째 fitness = {fitnesses[i]}')

        new_population = []
        for i in range(population_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            if generation == 0:
                print(f'{i+1}번째 child1 = {child1}, child2 = {child2}')
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    # 가장 적합한 개체 찾기
    best_fitness = max(fitnesses)
    best_index = fitnesses.index(best_fitness)
    best_individual = population[best_index]
    return best_individual, best_fitness




# 사용자로부터 목표 가격 입력 받기
target_price = int(input("목표 가격을 입력하세요: (만원)"))


# 유전 알고리즘 실행하여 최적의 구성 찾기
best_configuration, best_fitness = run_ga(items, target_price, population_size=100, generations=50)


# 결과 출력
print(f"가장 적합한 구성의 적응도: {best_fitness}")
print(f"가장 적합한 구성의 가격: {sum(best_configuration)}")
print(f"CPU 세대: {best_configuration[0] // 100}, RAM 크기: {best_configuration[1] // 10}, 스토리지 크기: {best_configuration[2] * 2}, GPU 크기: {best_configuration[3] // 20}")