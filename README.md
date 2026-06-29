# Monte Carlo Laser-based Robot Localization

Atividade desenvolvida para a disciplina **Introdução à Robótica Móvel**, semestre **2026.1**, do curso de **Engenharia de Computação** do **Instituto de Computação da UFAL**.

## Descrição

Este projeto implementa um algoritmo de localização probabilística para um robô móvel utilizando o método de **Monte Carlo Localization**, baseado em **filtro de partículas**.

O robô estima sua pose ((x, y, \theta)) em um mapa conhecido a partir da combinação de:

* odometria ruidosa;
* leituras do sensor laser;
* simulação de laser virtual por *ray casting*;
* atualização de pesos das partículas;
* reamostragem sistemática.

## Funcionamento

O algoritmo executa o seguinte ciclo:

1. inicializa partículas aleatoriamente em regiões livres do mapa;
2. propaga as partículas com base na odometria;
3. simula leituras de laser para cada partícula;
4. compara o laser simulado com o laser real;
5. atualiza os pesos das partículas;
6. realiza reamostragem quando necessário;
7. estima a pose final a partir do agrupamento de partículas mais provável.

## Principais parâmetros

* `N_PARTICLES = 300`: número de partículas utilizadas;
* `LASER_NUM_BEAMS = 24`: quantidade de feixes do laser considerados;
* `RESAMPLE_NEFF_RATIO = 0.60`: limiar para reamostragem;
* `INIT_SPIN_UPDATES = 25`: fase inicial de rotação para coleta de observações;
* `LOOP_HZ = 20`: frequência de execução do laço principal.


## Autoria

Rita de Kassia Lemos Pereira
Instituto de Computação — UFAL
2026.1
