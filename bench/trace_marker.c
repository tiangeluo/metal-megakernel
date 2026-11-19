#include <os/log.h>
#include <os/signpost.h>

// A separate diagnostic process brackets one synchronous step in Instruments.
static os_log_t log_handle;
static os_signpost_id_t interval_id;
void decode_marker_begin(const char *schedule) {
    log_handle = os_log_create("org.megakernel.benchmark", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    interval_id = os_signpost_id_generate(log_handle);
    os_signpost_interval_begin(log_handle, interval_id, "Decode step", "%{public}s", schedule);
}
void decode_marker_end(void) {
    os_signpost_interval_end(log_handle, interval_id, "Decode step");
}
