#ifndef BSH_GUMBEL_TEMPERATURE_H
#define BSH_GUMBEL_TEMPERATURE_H

typedef enum {
    TEMP_FIXED,
    TEMP_LINEAR,
    TEMP_EXPONENTIAL,
    TEMP_INVERSE_LOGISTIC
} temperature_schedule_t;

typedef struct {
    temperature_schedule_t schedule_type;
    float tau_max;
    float tau_min;
    int max_epochs;
    int current_epoch;
} temperature_schedule_t;

typedef struct {
    temperature_schedule_t schedule_type;
    float tau_max;
    float tau_min;
    int max_epochs;
    int current_epoch;
} temp_schedule_t;

temp_schedule_t* temp_schedule_create(
    temperature_schedule_t type,
    float tau_max,
    float tau_min,
    int max_epochs
);

void temp_schedule_free(temp_schedule_t* sched);

float temp_schedule_get(const temp_schedule_t* sched, int epoch);
float temp_schedule_current(const temp_schedule_t* sched);
void temp_schedule_step(temp_schedule_t* sched);

float temp_linear(float tau_max, float tau_min, int epoch, int max_epochs);
float temp_exponential(float tau_max, float tau_min, int epoch, int max_epochs);
float temp_inverse_logistic(float tau_max, float tau_min, int epoch, int max_epochs);

#endif
