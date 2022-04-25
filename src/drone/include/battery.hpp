#include <chrono>
#include <algorithm>

class Battery
{
public:
    Battery(): duration(0) {  }
    Battery(float duration) : duration(duration)
    {
        start = std::chrono::system_clock::now();
    }
    ~Battery() { }
    
    // battery duration in seconds
    const float duration;
    // function to publish battery remaining time
    float remainingTime()
    {
        auto now = std::chrono::system_clock::now();
        float elapsed_seconds = (now - start).count();

        float remaining_time = duration - elapsed_seconds;
        return std::max(remaining_time, 0.0f);
    }

    void refillBattery()
    {
        start = std::chrono::system_clock::now();
    }

private:
    std::chrono::_V2::system_clock::time_point start;
};