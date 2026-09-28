#include "temperature.h"
#include <stdlib.h>
#include <math.h>

float temp_linear(float tau_max, float tau_min, int epoch, int max_epochs) {
    if (max_epochs <= 0) return tau_min;
    float t = (float)epoch / (float)max_epochs;
    return tau_max + t * (tau_min - tau_max);
}

float temp_exponential(float tau_max, float tau_min, int epoch, int max_epochs) {
    if (max_epochs <= 0) return tau_min;
    float t = (float)epoch / (float)max_epochs;
    float rate = logf(tau_min / tau_max) / max_epochs;
    return tau_max * expf(rate * epoch);
}

float temp_inverse_logistic(float tau_max, float tau_min, int epoch, int max_epochs) {
    if (max_epochs <= 0) return tau_min;
    float t = (float)epoch / (float)max_epochs;
    float sigmoid = 1.0f / (1.0f + expf(-10.0f * (t - 0.5f)));
    return tau_min + (tau_max - tau_min) * (1.0f - sigmoid);
}

temp_schedule_t* temp_schedule_create(
    temperature_schedule_t type,
    float tau_max,
    float tau_min,
    int max_epochs
) {
    temp_schedule_t* sched = (temp_schedule_t*)malloc(sizeof(temp_schedule_t));
    if (sched) {
        sched->schedule_type = type;
        sched->tau_max = tau_max > 0.0f ? tau_max : 1.0f;
        sched->tau_min = tau_min > 0.0f ? tau_min : 0.1f;
        sched->max_epochs = max_epochs > 0 ? max_epochs : 1000;
        sched->current_epoch = 0;
    }
    return sched;
}

void temp_schedule_free(temp_schedule_t* sched) {
    if (sched) free(sched);
}

float temp_schedule_get(const temp_schedule_t* sched, int epoch) {
    if (!sched) return 1.0f;

    switch (sched->schedule_type) {
        case TEMP_FIXED:
            return sched->tau_max;
        case TEMP_LINEAR:
            return temp_linear(sched->tau_max, sched->tau_min, epoch, sched->max_epochs);
        case TEMP_EXPONENTIAL:
            return temp_exponential(sched->tau_max, sched->tau_min, epoch, sched->max_epochs);
        case TEMP_INVERSE_LOGISTIC:
            return temp_inverse_logistic(sched->tau_max, sched->tau_min, epoch, sched->max_epochs);
        default:
            return sched->tau_max;
    }
}

float temp_schedule_current(const temp_schedule_t* sched) {
    if (!sched) return 1.0f;
    return temp_schedule_get(sched, sched->current_epoch);
}

void temp_schedule_step(temp_schedule_t* sched) {
    if (sched && sched->current_epoch < sched->max_epochs) {
        sched->current_epoch++;
    }
}
