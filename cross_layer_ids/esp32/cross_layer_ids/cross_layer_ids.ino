/*
 * Cross-Layer IDS — ESP32 DevKit V1
 * VERIFIED FIRMWARE — Matches Python pipeline exactly
 *
 * Protocol IN:  S,f0,...,f6|C,f0,...,f6|V,f0,...,f4\n
 * Protocol OUT: R,class,feat_us,infer_us,xl0,...,xl5\n
 *
 * Computes 6 cross-layer physics features ON DEVICE:
 *   F19: Speed consistency (GPS vs CAN)
 *   F20: Yaw consistency (Bicycle model vs GPS)
 *   F21: Yaw consistency (Bicycle model vs IMU)
 *   F22: Lateral acceleration (GPS-derived vs IMU)
 *   F23: Obstacle consistency (Ultrasonic vs V2X)
 *   F24: 3-way curvature (V2X vs CAN vs GPS)
 *
 * Wiring:
 *   GPIO2 -> 220ohm -> Green LED -> GND
 *   GPIO4 -> 220ohm -> Yellow LED -> GND
 *   GPIO5 -> 220ohm -> Red LED -> GND
 */

#include <Arduino.h>
#include <math.h>
#include "rf_model.h"

// ── Pins ──
#define PIN_GREEN   2
#define PIN_YELLOW  4
#define PIN_RED     5

// ── Constants ──
#define WHEELBASE   2.7f
#define NFEAT       25
#define EPS         1e-6f
#define BUFSIZE     512

// ── Buffers ──
char msg_buf[BUFSIZE];
int msg_idx = 0;

// ── Data ──
float sd[7]    = {0};  // sensor data
float cd[7]    = {0};  // CAN data
float vd[5]    = {0};  // V2X data
float feat[NFEAT] = {0};  // all 25 features

// ── Timing ──
unsigned long tFeat  = 0;
unsigned long tInfer = 0;
uint32_t sampleCount = 0;
uint32_t classCount[3] = {0, 0, 0};


// ══════════════════════════════════════════════════════════
// CROSS-LAYER PHYSICS COMPUTATION
// Formulas match Python compute_cross_layer() exactly
// ══════════════════════════════════════════════════════════

void computeFeatures() {
    unsigned long t0 = micros();

    // Copy raw features into feature array
    for (int i = 0; i < 7; i++) feat[i]     = sd[i];
    for (int i = 0; i < 7; i++) feat[7 + i]  = cd[i];
    for (int i = 0; i < 5; i++) feat[14 + i] = vd[i];

    // Aliases for readability
    float gps_speed   = feat[0];
    float gps_heading = feat[1];
    float imu_lat     = feat[2];
    float imu_yaw     = feat[3];
    float ultra_min   = feat[5];
    float can_speed   = feat[7];
    float can_steer   = feat[8];
    float v2x_curv    = feat[14];
    float v2x_obs     = feat[16];

    // ── F19: Speed Consistency (GPS vs CAN) ──
    // Python: abs(gps_speed - wheel_speed) / max(abs(both)) + eps
    float max_spd = fmaxf(fabsf(gps_speed), fabsf(can_speed)) + EPS;
    feat[19] = fminf(fabsf(gps_speed - can_speed) / max_spd, 1.0f);

    // ── F20: Yaw CAN vs GPS ──
    // Python: bicycle model yaw = (speed * tan(steer)) / wheelbase
    float yaw_can = (can_speed * tanf(can_steer)) / WHEELBASE;
    float max_y1 = fmaxf(fabsf(yaw_can), fabsf(gps_heading)) + EPS;
    feat[20] = fminf(fabsf(yaw_can - gps_heading) / max_y1, 1.0f);

    // ── F21: Yaw CAN vs IMU ──
    // Python: same bicycle yaw vs IMU gyroscope
    float max_y2 = fmaxf(fabsf(yaw_can), fabsf(imu_yaw)) + EPS;
    feat[21] = fminf(fabsf(yaw_can - imu_yaw) / max_y2, 1.0f);

    // ── F22: Lateral Acceleration (GPS-derived vs IMU) ──
    // Python: a_lat = gps_speed * gps_heading_rate (centripetal)
    float gps_lat = gps_speed * gps_heading;
    float max_lat = fmaxf(fabsf(gps_lat), fabsf(imu_lat)) + EPS;
    feat[22] = fminf(fabsf(gps_lat - imu_lat) / max_lat, 1.0f);

    // ── F23: Obstacle Consistency (Ultrasonic vs V2X) ──
    // Python: only compute when V2X reports nearby obstacle
    if (v2x_obs < 50.0f) {
        float max_obs = fmaxf(ultra_min, v2x_obs) + EPS;
        feat[23] = fminf(fabsf(ultra_min - v2x_obs) / max_obs, 1.0f);
    } else {
        feat[23] = 0.0f;
    }

    // ── F24: 3-Way Curvature Consistency ──
    // Python: 0.5 * (|v2x - can| / max(v2x,can) + |v2x - gps| / max(v2x,gps))
    float curv_can = tanf(can_steer) / WHEELBASE;
    float curv_gps = gps_heading / (gps_speed + EPS);

    float max_cv = fmaxf(fabsf(v2x_curv), fabsf(curv_can)) + EPS;
    float max_gv = fmaxf(fabsf(v2x_curv), fabsf(curv_gps)) + EPS;
    feat[24] = fminf(
        0.5f * (fabsf(v2x_curv - curv_can) / max_cv +
                fabsf(v2x_curv - curv_gps) / max_gv),
        1.0f
    );

    tFeat = micros() - t0;
}


// ══════════════════════════════════════════════════════════
// RANDOM FOREST INFERENCE
// Uses arrays from rf_model.h
// ══════════════════════════════════════════════════════════

int predTree(int ti) {
    int node = 0;
    for (int d = 0; d < 25; d++) {  // depth limit for safety
        int16_t fi = get_feat(ti, node);
        if (fi < 0) {
            // Leaf node — return class
            return get_cls(ti, node);
        }
        if (fi >= NFEAT) return 0;  // safety check

        if (feat[fi] <= get_thr(ti, node)) {
            node = get_left(ti, node);
        } else {
            node = get_right(ti, node);
        }
        if (node < 0) return 0;  // safety check
    }
    return 0;  // fallback
}


int predRF() {
    unsigned long t0 = micros();
    int votes[3] = {0, 0, 0};

    for (int t = 0; t < RF_N_TREES; t++) {
        int cls = predTree(t);
        if (cls >= 0 && cls < 3) {
            votes[cls]++;
        }
    }

    tInfer = micros() - t0;

    // Majority vote
    int best = 0;
    for (int c = 1; c < 3; c++) {
        if (votes[c] > votes[best]) best = c;
    }
    return best;
}


// ══════════════════════════════════════════════════════════
// SERIAL PROTOCOL PARSER
// Format: S,f0,f1,...,f6|C,f0,...,f6|V,f0,...,f4\n
// Uses char pointers and strtof — no String objects
// ══════════════════════════════════════════════════════════

bool pFloat(const char* str, float* out, int count) {
    for (int i = 0; i < count; i++) {
        char* end;
        out[i] = strtof(str, &end);
        if (end == str) return false;  // parse failed
        str = end;
        if (*str == ',') str++;  // skip comma
    }
    return true;
}


bool parsePkt(const char* line) {
    // Find the two '|' separators
    const char* sep1 = strchr(line, '|');
    if (!sep1) return false;
    const char* sep2 = strchr(sep1 + 1, '|');
    if (!sep2) return false;

    // Parse sensor block: S,f0,f1,...,f6
    if (line[0] != 'S' || line[1] != ',') return false;
    if (!pFloat(line + 2, sd, 7)) return false;

    // Parse CAN block: C,f0,f1,...,f6
    if (sep1[1] != 'C' || sep1[2] != ',') return false;
    if (!pFloat(sep1 + 3, cd, 7)) return false;

    // Parse V2X block: V,f0,f1,...,f4
    if (sep2[1] != 'V' || sep2[2] != ',') return false;
    if (!pFloat(sep2 + 3, vd, 5)) return false;

    return true;
}


// ══════════════════════════════════════════════════════════
// LED OUTPUT
// ══════════════════════════════════════════════════════════

void setLEDs(int prediction) {
    digitalWrite(PIN_GREEN,  prediction == 0 ? HIGH : LOW);
    digitalWrite(PIN_YELLOW, prediction == 1 ? HIGH : LOW);
    digitalWrite(PIN_RED,    prediction == 2 ? HIGH : LOW);
}


// ══════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);

    pinMode(PIN_GREEN,  OUTPUT);
    pinMode(PIN_YELLOW, OUTPUT);
    pinMode(PIN_RED,    OUTPUT);

    // Boot sequence: cycle LEDs
    int pins[] = {PIN_GREEN, PIN_YELLOW, PIN_RED};
    for (int i = 0; i < 3; i++) {
        digitalWrite(pins[i], HIGH);
        delay(150);
        digitalWrite(pins[i], LOW);
    }
    // Flash all on briefly
    for (int i = 0; i < 3; i++) digitalWrite(pins[i], HIGH);
    delay(300);
    for (int i = 0; i < 3; i++) digitalWrite(pins[i], LOW);

    Serial.println("# CrossLayer-IDS v2.0 Ready");
    Serial.printf("# Trees=%d Features=%d\n", RF_N_TREES, RF_N_FEATURES);
}


// ══════════════════════════════════════════════════════════
// MAIN LOOP
// Character-by-character serial reading into buffer
// Processes complete lines ending with \n
// ══════════════════════════════════════════════════════════

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            // Process complete line
            if (msg_idx > 20) {  // minimum valid packet
                msg_buf[msg_idx] = '\0';

                if (parsePkt(msg_buf)) {
                    // Compute cross-layer features on device
                    computeFeatures();

                    // Run Random Forest inference
                    int prediction = predRF();

                    // Update LEDs
                    setLEDs(prediction);

                    // Send result in standard format
                    // R,class,feat_us,infer_us,xl0,xl1,xl2,xl3,xl4,xl5
                    Serial.printf(
                        "R,%d,%lu,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        prediction,
                        tFeat,
                        tInfer,
                        feat[19], feat[20], feat[21],
                        feat[22], feat[23], feat[24]
                    );

                    // Statistics
                    sampleCount++;
                    if (prediction >= 0 && prediction < 3) {
                        classCount[prediction]++;
                    }

                    // Periodic debug output
                    if (sampleCount % 200 == 0) {
                        Serial.printf(
                            "# [%lu] N=%lu S=%lu C=%lu "
                            "feat=%luus infer=%luus\n",
                            sampleCount,
                            classCount[0],
                            classCount[1],
                            classCount[2],
                            tFeat,
                            tInfer
                        );
                    }
                }
            }
            msg_idx = 0;  // reset buffer

        } else if (msg_idx < BUFSIZE - 2) {
            // Add character to buffer
            msg_buf[msg_idx++] = c;

        } else {
            // Buffer overflow protection — reset
            msg_idx = 0;
        }
    }
}
