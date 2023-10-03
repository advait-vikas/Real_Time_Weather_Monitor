#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

double compute_temperature(int location) {
    double base_temperature = location * 2.0;
    double random_multiplier = ((double)rand() / RAND_MAX) * 0.4 + 0.8;
    double weather = base_temperature * random_multiplier;
    return weather;
}

const char* get_weather_condition(double temperature) {
    if (temperature > 30.0) {
        return "Sunny";
    } else if (temperature > 20.0) {
        return "Partly Cloudy";
    } else {
        return "Rainy";
    }
}

double calculate_variance(double* temperatures, int num_temperatures, double average) {
    double sum = 0.0;

    for (int i = 0; i < num_temperatures; i++) {
        double diff = temperatures[i] - average;
        sum += diff * diff;
    }

    return sum / num_temperatures;
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    if (rank == 0) {
        printf("Weather Forecasting using Open MPI\n");
    }

    char* locations[] = {
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Miami",
        "San Francisco",
        "Boston",
        "Seattle",
        "Denver",
        "Dallas"
    };
    int num_locations = sizeof(locations) / sizeof(locations[0]);

    int chunk_size = num_locations / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;

    if (rank == size - 1) {
        end = num_locations;
    }

    double* temperatures = (double*)malloc((end - start) * sizeof(double));
    double total_temperature = 0.0;
    double max_temperature = -9999.0;
    double min_temperature = 9999.0;

    for (int i = start; i < end; i++) {
        double temperature = compute_temperature(i);
        temperatures[i - start] = temperature;
        total_temperature += temperature;

        if (temperature > max_temperature) {
            max_temperature = temperature;
        }

        if (temperature < min_temperature) {
            min_temperature = temperature;
        }

        const char* condition = get_weather_condition(temperature);
        printf("Process %d forecasted %s weather for location %s: %.2f°C\n", rank, condition, locations[i], temperature);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double average_temperature = total_temperature / (end - start);
    printf("Process %d average temperature for its locations: %.2f°C\n", rank, average_temperature);

    double global_average_temperature;
    MPI_Allreduce(&average_temperature, &global_average_temperature, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        global_average_temperature /= size;
        printf("Global average temperature across all processes: %.2f°C\n", global_average_temperature);
    }

    double global_max_temperature;
    MPI_Reduce(&max_temperature, &global_max_temperature, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Maximum temperature across all locations and processes: %.2f°C\n", global_max_temperature);
    }

    double global_min_temperature;
    MPI_Reduce(&min_temperature, &global_min_temperature, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Minimum temperature across all locations and processes: %.2f°C\n", global_min_temperature);
    }

    double variance = calculate_variance(temperatures, end - start, average_temperature);
    printf("Process %d variance of temperatures for its locations: %.2f\n", rank, variance);

    double global_variance;
    MPI_Allreduce(&variance, &global_variance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        global_variance /= size;
        printf("Global variance of temperatures across all processes: %.2f\n", global_variance);
    }

    free(temperatures);
    MPI_Finalize();
}

