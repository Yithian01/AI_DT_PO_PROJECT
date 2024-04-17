import random
import pandas as pd
# 부품별 가격 범위 설정
cpu_prices = {generation: 100 * generation for generation in range(1, 11)}  # 1세대부터 10세대까지 
ram_prices = {size: 10 * size for size in [4, 8, 16, 32, 64]}  # 4GB부터 64GB까지 
storage_prices = {size: 0.5 * size for size in [128, 256, 512, 1024, 2048]}  # 128GB부터 2TB까지 
gpu_prices = {size: 20 * size for size in [2, 4, 8, 16]}  # 2GB부터 16GB까지 



print(f'cpu_prices = {cpu_prices}')
print(f'ram_prices = {ram_prices}')
print(f'storage_prices = {storage_prices}')
print(f'gpu_prices = {gpu_prices}')

# 데이터셋 생성 함수 ----> 필요없음
def generate_dataset(n):
    dataset = []
    for _ in range(n):
        cpu_gen = random.randint(1, 10)
        ram_size = random.choice([4, 8, 16, 32, 64])
        storage_size = random.choice([128, 256, 512, 1024, 2048])
        gpu_size = random.choice([2, 4, 8, 16])

        price = cpu_prices[cpu_gen] + ram_prices[ram_size] + storage_prices[storage_size] + gpu_prices[gpu_size]

        dataset.append({
            "cpu": cpu_gen,
            "ram": ram_size,
            "storage": storage_size,
            "gpu": gpu_size,
            "price": price
        })
    return dataset
# 예제 데이터셋 생성 -- 초기 개체군 
#dataset = generate_dataset(100)
#print(dataset[:5])  # 처음 5개의 데이터를 출력


'''적응도 함수'''
def fitness(individual, target):
    # 개체의 가격을 계산
    price = sum(individual)
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


# 부품별 가격 범위 설정
items = [
    list(cpu_prices.values()),  # CPU 가격
    list(ram_prices.values()),  # RAM 가격
    list(storage_prices.values()),  # 스토리지 가격
    list(gpu_prices.values())  # GPU 가격
]


# 사용자로부터 목표 가격 입력 받기
target_price = int(input("목표 가격을 입력하세요: (만원)"))


# 유전 알고리즘 실행하여 최적의 구성 찾기
best_configuration, best_fitness = run_ga(items, target_price, population_size=100, generations=50)


# 결과 출력
print(f"가장 적합한 구성의 적응도: {best_fitness}")
print(f"가장 적합한 구성의 가격: {sum(best_configuration)}")
print(f"CPU 세대: {best_configuration[0] // 100}, RAM 크기: {best_configuration[1] // 10}, 스토리지 크기: {best_configuration[2] * 2}, GPU 크기: {best_configuration[3] // 20}")