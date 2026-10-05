# Пример вывода программы

Команда:

```bash
printf '0\nwip\nq\nn\n' | ./build/src/ds_load --dataset data/garments_worker_productivity.csv
```

Ввод пользователя: `0` (номер признака `quarter`), `wip` (имя признака),
`q` (выход из цикла статистики), `n` (не сохранять преобразованный набор).

## Вывод

```text
Objects: 1197
Features: 14

    # | Feature                   | Type                 |         Missing
---------------------------------------------------------------------------
    0 | quarter                   | categorical          |               0
    1 | department                | categorical          |               0
    2 | day                       | categorical          |               0
    3 | team                      | numeric              |               0
    4 | targeted_productivity     | numeric              |               0
    5 | smv                       | numeric              |               0
    6 | wip                       | numeric              |    506 (42.27%)
    7 | over_time                 | numeric              |               0
    8 | incentive                 | numeric              |               0
    9 | idle_time                 | numeric              |               0
   10 | idle_men                  | numeric              |               0
   11 | no_of_style_change        | categorical          |               0
   12 | no_of_workers             | numeric              |               0
   13 | actual_productivity       | numeric              |               0
Total: 506 missing values in 1 distinct features.Database Statictics.

Print the column number or column name of the feature(or s/summary to print summary or q/quit to exit statistics): 
Feature 0:    quarter (categorical)

   Missing values: 0
   Categories list: Quarter1; Quarter2; Quarter3; Quarter4; Quarter5;
   Quarter5            #############                                                                                          44
   Quarter4            #####################################################################                                  248
   Quarter3            ###########################################################                                            210
   Quarter2            ##############################################################################################         335
   Quarter1            ####################################################################################################   360
Print the column number or column name of the feature(or s/summary to print summary or q/quit to exit statistics): 
Feature 6:        wip (numeric)

   Missing values: 506
   min: 7.000     max: 23122.000
   mean: 1190.466     var: 1836.125

   q05: 358.500
   q25: 774.500
   q50: 1039.000
   q75: 1252.500
   q95: 1602.000

   [7.0, 4630.0)       ####################################################################################################   683
   [4630.0, 9253.0)    #                                                                                                      1
   [9253.0, 13876.0)   #                                                                                                      2
   [13876.0, 18499.0)  #                                                                                                      1
   [18499.0, 23122.0)  #                                                                                                      4
Print the column number or column name of the feature(or s/summary to print summary or q/quit to exit statistics): 
Save the transformed dataset?(y/yes, n/no): 
```

## Пример для признака с числовыми пропусками и для категориального
признака, попавшего под порог

Команда:

```bash
printf '3\nno_of_style_change\nq\nn\n' | ./build/src/ds_load --dataset data/garments_worker_productivity.csv
```

```text
Feature 3:       team (numeric)

   Missing values: 0
   min: 1.000     max: 12.000
   mean: 6.427     var: 3.463

   q05: 1.000
   q25: 3.000
   q50: 6.000
   q75: 9.000
   q95: 12.000

   [1.0, 3.2)          ####################################################################################################   309
   [3.2, 5.4)          #################################################################                                      198
   [5.4, 7.6)          ##############################################################                                         190
   [7.6, 9.8)          #####################################################################                                  213
   [9.8, 12.0)         #############################################################################################          287

Feature 11: no_of_style_change (categorical)

   Missing values: 0
   Categories list: 0.000000; 1.000000; 2.000000;
   1.000000            ###########                                                                                            114
   2.000000            ####                                                                                                   33
   0.000000            ####################################################################################################   1050
```

## Параметры командной строки

```text
$ ./build/src/ds_load --help
Load dataset from file and transform it into numeric matrix.
Usage:
ds_load [OPTION...]

  -h, --help           Help
  -f, --csv arg        Path to the dataset
  -o, --dest arg       Path to the file with processed data
      --histogram arg  Number of histogram intervals
  -i, --fill arg       Imputation strategy (default: mean)
  -s, --scale arg      Scaling strategy (default: minmax)
  -e, --encode arg     Encoding strategy (default: onehot)
```
