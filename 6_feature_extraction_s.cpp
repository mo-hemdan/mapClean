#include <iostream>
// #include <arrow/api.h>
// #include <parquet/arrow/reader.h>
// #include <parquet/file_reader.h>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <list>
#include "Profiler.h"
// #include <set>

// #include <sys/resource.h>
// #include <sys/time.h>
using namespace std;

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;
using namespace std::chrono;
using Duration = duration<double>;

#include <chrono>

string CITY;

// Indexing
int SUPER_POINT_SIZE;
int CELL_WIDTH;

// Initial Perfect Match Extraction
float DELTA_O;
float BETA_O;

// Initial Error Injection
float GAMMA_O;
int MAX_ROAD_LENGTH_O;
int MU_O;
int SIGMA_O;
float P_NOISE_O;
bool REMOVAL_ROADS_GROUPING_O, REMOVAL_ROAD_MAXLENGTH_OPTION_O;

// Perfect Match Extraction
float DELTA, BETA;

// Error Injection
float GAMMA; // std::round((GAMMA_O / (P_NOISE_O * (1.0 - GAMMA_O) + GAMMA_O)) * 100.0) / 100.0;
int MAX_ROAD_LENGTH, MU, SIGMA, P_NOISE;
bool REMOVAL_ROADS_GROUPING, REMOVAL_ROAD_MAXLENGTH_OPTION;

// Feature Extraction
int D, N;

int MAX_ROWS, FEATURE_MAX_ROWS, N_CELLS_X, N_CELLS_Y;
double CELL_WIDTH_X, CELL_WIDTH_Y;

// Input Dataset Storage
tuple<int, int> *geoms;
float *speeds, *distance_to_road, *new_distance_to_road, *new_r_p_sim;
int *angles, *n_points, *new_road_angle;
bool *to_be_matched;
tuple<long long, long long, short> *road_ids, *new_road_ids;
string *types;

// Features
double *W_c, *W_s, *W_ms, *W_as, *W_ss, *W_r, *W_Dir;
float *N_c, *N_s, *N_ms, *N_as, *N_ss;
int *N_r, *N_Dir;

// Index
int ***Cell2PointIndex;
int **Cell2PointsSize;

bool load_config(const string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Could not open config file: " << filename << "\n";
        return false;
    }

    json j;
    file >> j;

    // Load base values
    CITY = j["CITY"];
    SUPER_POINT_SIZE = j["SUPER_POINT_SIZE"];
    CELL_WIDTH = j["CELL_WIDTH"];

    DELTA_O = j["DELTA_O"];
    BETA_O = j["BETA_O"];
    GAMMA_O = j["GAMMA_O"];
    GAMMA = j["GAMMA"];
    MAX_ROAD_LENGTH_O = j["MAX_ROAD_LENGTH_O"];
    MU_O = j["MU_O"];
    SIGMA_O = j["SIGMA_O"];
    SIGMA = j["SIGMA"];
    P_NOISE_O = j["P_NOISE_O"];

    REMOVAL_ROADS_GROUPING_O = j["REMOVAL_ROADS_GROUPING_O"];
    REMOVAL_ROAD_MAXLENGTH_OPTION_O = j["REMOVAL_ROAD_MAXLENGTH_OPTION_O"];

    D = j["D"];
    N = j["N"];

    return true;
}

// Function to read the JSON file
void updateGridInfo(const std::string &filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    json j;
    f >> j;

    CELL_WIDTH_X = j.at("cell_width_x").get<double>();
    CELL_WIDTH_Y = j.at("cell_width_y").get<double>();
    MAX_ROWS = j.at("total_points").get<int>();
    FEATURE_MAX_ROWS = j.at("non_old_points").get<int>();
    N_CELLS_X = j.at("n_cells_x").get<int>();
    N_CELLS_Y = j.at("n_cells_y").get<int>();
}

void update_progress(int current, int total, int bar_width, steady_clock::time_point start_time)
{
    float progress = (float)current / total;
    int pos = bar_width * progress;

    // Time elapsed
    auto now = steady_clock::now();
    auto elapsed = duration_cast<seconds>(now - start_time).count();

    cout << "[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% ";
    cout << "Elapsed: " << elapsed << "s\r";
    cout.flush();
}

// Function to convert a string key like "[33, 33]" into a tuple<int, int>
tuple<int, int> parseTupleKey(const string &key)
{
    int num1, num2;
    char ignore; // To ignore ',' and brackets

    stringstream ss(key);
    ss >> ignore >> num1 >> ignore >> num2 >> ignore; // Read "[33, 33]"

    return make_tuple(num1, num2);
}
// Custom hash function for tuple<int, int>
struct TupleHash
{
    size_t operator()(const tuple<int, int> &t) const
    {
        return hash<int>()(get<0>(t)) ^ (hash<int>()(get<1>(t)) << 1);
    }
};

struct TupleHash3
{
    template <class T>
    inline void hash_combine(std::size_t &seed, const T &val) const
    {
        seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    size_t operator()(const std::tuple<long long, long long, short> &t) const noexcept
    {
        size_t seed = 0;
        hash_combine(seed, std::get<0>(t));
        hash_combine(seed, std::get<1>(t));
        hash_combine(seed, std::get<2>(t));
        return seed;
    }
};

unordered_map<tuple<int, int>, vector<int>, TupleHash> load_json(const string &filename)
{
    unordered_map<tuple<int, int>, vector<int>, TupleHash> data;
    ifstream file(filename);
    if (!file)
    {
        cerr << "Error opening file!\n";
        return data;
    }

    json j;
    file >> j;

    for (auto &[key, value] : j.items())
    {
        tuple<int, int> key_int = parseTupleKey(key); // Convert string keys to int
        vector<int> vec = value.get<vector<int>>();
        data[key_int] = vec;
    }

    return data;
}

bool str_to_bool(const string &s)
{
    return s == "true"; // case-sensitive
}

int read_gdf(string filename, bool prnt = false)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file\n";
        return 1;
    }

    string line;
    getline(file, line); // skip header

    int row = 0;
    long long a, b;
    short c;
    char ignore;
    while (getline(file, line) && row < MAX_ROWS)
    {
        stringstream ss(line);
        string cell;

        // Column 1: lat
        getline(ss, cell, ',');
        int lat = stoi(cell);

        // Column 2: lon
        getline(ss, cell, ',');
        int lon = stoi(cell);

        geoms[row] = make_tuple(lon, lat);

        // Column 3: speed
        getline(ss, cell, ',');
        float speed = stof(cell);
        speeds[row] = speed;

        // Column 4: angle
        getline(ss, cell, ',');
        int angle = stoi(cell);
        angles[row] = angle;

        // Column 5: nPoints
        getline(ss, cell, ',');
        int nPoints = stoi(cell);
        n_points[row] = nPoints;

        // // Column 6: ture_toBeMatched
        // getline(ss, cell, ',');
        // ture_to_be_matched[row] = str_to_bool(cell);

        // Column 7: matched_road_id
        getline(ss, cell, ',');
        stringstream tuple_stream(cell);
        tuple_stream >> ignore >> a >> b >> c >> ignore;
        road_ids[row] = make_tuple(a, b, c);

        // Column 8: distance_to_matched_road
        getline(ss, cell, ',');
        float distance = stof(cell);
        distance_to_road[row] = distance;

        // Column 9: type
        getline(ss, cell, ',');
        types[row] = cell;

        // Column 10: toBeMatched
        getline(ss, cell, ',');
        to_be_matched[row] = str_to_bool(cell);

        // Column 11: new_matched_road_id
        getline(ss, cell, ',');
        stringstream new_tuple_stream(cell);
        new_tuple_stream >> ignore >> a >> b >> c >> ignore;
        new_road_ids[row] = make_tuple(a, b, c);

        // Column 12: new_distance_to_matched_road
        getline(ss, cell, ',');
        distance = stof(cell);
        new_distance_to_road[row] = distance;

        // Column 13: new_road_angle
        getline(ss, cell, ',');
        int road_a = stof(cell);
        new_road_angle[row] = road_a;

        // Column 14: new_r_p_sim
        getline(ss, cell, ',');
        float r_p_s = stof(cell);
        new_r_p_sim[row] = r_p_s;

        if (prnt)
            cout << "Output [" << row << "]: " << "("
                 << get<0>(geoms[row]) << ", "
                 << get<1>(geoms[row]) << "), "
                 << speeds[row] << ", "
                 << angles[row] << ", "
                 << n_points[row] << ", ("
                 << get<0>(road_ids[row]) << ", "
                 << get<1>(road_ids[row]) << ", "
                 << get<2>(road_ids[row]) << "), "
                 << distance_to_road[row] << ", "
                 << types[row] << ", "
                 << to_be_matched[row] << ", ("
                 << get<0>(new_road_ids[row]) << ", "
                 << get<1>(new_road_ids[row]) << ", "
                 << get<2>(new_road_ids[row]) << "), "
                 << new_distance_to_road[row]
                 << endl;
        row++;
    }
    return 0;
}

bool compare_cell(tuple<int, int> &cell1, tuple<int, int> &cell2)
{
    if (get<0>(cell1) == get<0>(cell2))
    {
        return get<1>(cell1) < get<1>(cell2);
    }
    return get<0>(cell1) < get<0>(cell2);
}

// Calculate the manhattan distance
float calc_distance(tuple<int, int> &p1, tuple<int, int> &p2)
{
    // n_comparisions++;
    return abs(get<0>(p1) - get<0>(p2)) + abs(get<1>(p1) - get<1>(p2));
    // float dx = static_cast<float>(get<0>(p1) - get<0>(p2));
    // float dy = static_cast<float>(get<1>(p1) - get<1>(p2));
    // return sqrt(dx * dx + dy * dy);
}

bool is_forbidden(string &p_type, string &q_type)
{
    return (p_type == "old" && q_type == "bad") ||
           (p_type == "bad" && q_type == "old") ||
           (p_type == "old" && q_type == "good") ||
           (p_type == "good" && q_type == "old") ||
           (p_type == "old" && q_type == "old");
}

void print_duration_vector(const vector<Duration> &vec, const string &name)
{
    cout << name << ": [ ";
    for (const auto &dur : vec)
    {
        cout << duration_cast<milliseconds>(dur).count() << "ms ";
    }
    cout << "]" << endl;
}

int apply_override(int argc, char *argv[])
{
    // Check if the config file is provided
    if (argc < 2)
    {
        cout << "Error: Provide a config file. Example: ./program config.json\n";
        return 1;
    }
    // Load the config file
    string config_filename = argv[1];
    if (!load_config(config_filename))
        return 1;
    cout << "Loaded config file: " << config_filename << endl;

    if (argc % 2 != 0 && argc > 2)
    {
        cout << "Error: Provide variable-value pairs. Example: CITY Jakarta CELL_WIDTH 20\n";
        return 1;
    }

    for (int i = 2; i < argc; i += 2)
    {
        string var_name = argv[i];
        string var_value = argv[i + 1];

        if (var_name == "CITY")
            CITY = var_value;
        else if (var_name == "SUPER_POINT_SIZE")
            SUPER_POINT_SIZE = stoi(var_value);
        else if (var_name == "CELL_WIDTH")
            CELL_WIDTH = stoi(var_value);
        else if (var_name == "DELTA_O")
            DELTA_O = stoi(var_value);
        else if (var_name == "BETA_O")
            BETA_O = stod(var_value);
        else if (var_name == "GAMMA_O")
            GAMMA_O = stod(var_value);
        else if (var_name == "GAMMA")
            GAMMA = stod(var_value);
        else if (var_name == "MAX_ROAD_LENGTH_O")
            MAX_ROAD_LENGTH_O = stoi(var_value);
        else if (var_name == "MU_O")
            MU_O = stoi(var_value);
        else if (var_name == "SIGMA_O")
            SIGMA_O = stod(var_value);
        else if (var_name == "SIGMA")
            SIGMA = stod(var_value);
        else if (var_name == "P_NOISE_O")
            P_NOISE_O = stod(var_value);
        else if (var_name == "REMOVAL_ROAD_MAXLENGTH_OPTION_O")
            REMOVAL_ROAD_MAXLENGTH_OPTION_O = (var_value == "true");
        else if (var_name == "D")
            D = stoi(var_value);
        else if (var_name == "N")
            N = stoi(var_value);
        else
            cerr << "Unknown variable: " << var_name << endl;
    }
    return 0;
}

template <typename K, typename V, typename Hash = std::hash<K>>
size_t approxUnorderedMapMemory(const std::unordered_map<K, V, Hash> &m)
{
    size_t nodeOverhead = 3 * sizeof(void *);

    size_t total = sizeof(m);

    for (const auto &kv : m)
    {
        total += sizeof(kv.first) + sizeof(kv.second) + nodeOverhead;
    }

    return total;
}

template <typename K, typename V, typename Hash = std::hash<K>>
size_t approxUnorderedMapArrayMemory(std::unordered_map<K, V, Hash> *arr, size_t size)
{
    size_t total = sizeof(std::unordered_map<K, V, Hash>) * size;

    for (size_t i = 0; i < size; ++i)
    {
        total += approxUnorderedMapMemory(arr[i]);
    }

    return total;
}

// size_t getMemoryUsageKB() {
//     std::ifstream file("/proc/self/status");
//     std::string line;
//     while (std::getline(file, line)) {
//         if (line.rfind("VmRSS:", 0) == 0) { // Resident Set Size (RAM in KB)
//             size_t mem;
//             sscanf(line.c_str(), "VmRSS: %zu", &mem);
//             return mem; // KB
//         }
//     }
//     return 0;
// }

// double getCPUTime() {
//     struct rusage usage;
//     getrusage(RUSAGE_SELF, &usage);
//     return (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6) +
//            (usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6);
// }

string formatMemory(size_t kb)
{
    static const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    double size = static_cast<double>(kb);
    int unitIndex = 0;
    while (size >= 1024.0 && unitIndex < 3)
    {
        size /= 1024.0;
        unitIndex++;
    }
    ostringstream oss;
    oss << fixed << setprecision(2) << size << " " << units[unitIndex];
    return oss.str();
}

int main(int argc, char *argv[])
{
    if (apply_override(argc, argv))
        return 1;

    // Derived values
    DELTA = DELTA_O;
    BETA = BETA_O;
    // GAMMA = round(GAMMA_O / (P_NOISE_O * (1 - GAMMA_O) + GAMMA_O) * 100.0) / 100.0;
    MAX_ROAD_LENGTH = MAX_ROAD_LENGTH_O;
    MU = MU_O;
    // SIGMA = SIGMA_O;
    P_NOISE = 1.0;
    REMOVAL_ROADS_GROUPING_O = REMOVAL_ROAD_MAXLENGTH_OPTION_O; // ADDED FOR THE Sake they are similar
    REMOVAL_ROADS_GROUPING = REMOVAL_ROADS_GROUPING_O;
    REMOVAL_ROAD_MAXLENGTH_OPTION = REMOVAL_ROAD_MAXLENGTH_OPTION_O;

    ostringstream gdf_filename_stream;
    gdf_filename_stream << "city_" << CITY << "_pro/systemError_"
                        << "g" << static_cast<int>(100 * GAMMA)
                        << "s" << SIGMA
                        << "_PMEb" << static_cast<int>(100 * BETA)
                        << "g" << DELTA
                        << "_PreC" << CELL_WIDTH
                        << "sup" << SUPER_POINT_SIZE
                        << "_OERg" << static_cast<int>(100 * GAMMA_O)
                        << "s" << SIGMA_O
                        << "p" << static_cast<int>(100 * P_NOISE_O)
                        << "gr" << static_cast<int>(REMOVAL_ROADS_GROUPING_O)
                        << "mr" << static_cast<int>(REMOVAL_ROAD_MAXLENGTH_OPTION_O)
                        << ".csv";
    string GDF_FILENAME = gdf_filename_stream.str();
    cout << "GDF_FILENAME: " << GDF_FILENAME << endl;

    ostringstream index_filename_stream;
    index_filename_stream << "city_" << CITY
                          << "_pro/ErrorIndex_"
                          << "g" << static_cast<int>(100 * GAMMA)
                          << "s" << SIGMA
                          << "_PMEb" << static_cast<int>(100 * BETA)
                          << "g" << DELTA
                          << "_PreC" << CELL_WIDTH
                          << "sup" << SUPER_POINT_SIZE
                          << "_OERg" << static_cast<int>(100 * GAMMA_O)
                          << "s" << SIGMA_O
                          << "p" << static_cast<int>(100 * P_NOISE_O)
                          << "gr" << static_cast<int>(REMOVAL_ROADS_GROUPING_O)
                          << "mr" << static_cast<int>(REMOVAL_ROAD_MAXLENGTH_OPTION_O)
                          << ".json";
    string INDEX_FILENAME = index_filename_stream.str();
    cout << "INDEX_FILENAME: " << INDEX_FILENAME << endl;

    ostringstream format_filename_stream;
    format_filename_stream << "city_" << CITY
                           << "_pro/Format_"
                           << "g" << static_cast<int>(100 * GAMMA)
                           << "s" << SIGMA
                           << "_PMEb" << static_cast<int>(100 * BETA)
                           << "g" << DELTA
                           << "_PreC" << CELL_WIDTH
                           << "sup" << SUPER_POINT_SIZE
                           << "_OERg" << static_cast<int>(100 * GAMMA_O)
                           << "s" << SIGMA_O
                           << "p" << static_cast<int>(100 * P_NOISE_O)
                           << "gr" << static_cast<int>(REMOVAL_ROADS_GROUPING_O)
                           << "mr" << static_cast<int>(REMOVAL_ROAD_MAXLENGTH_OPTION_O)
                           << ".json";
    string FORMAT_FILENAME = format_filename_stream.str();
    cout << "FORMAT_FILENAME: " << FORMAT_FILENAME << endl;

    ostringstream output_filename_stream;
    output_filename_stream << "city_" << CITY
                           << "_pro/featureEx_d" << D
                           << "n" << N
                           << "_SERg" << static_cast<int>(100 * GAMMA)
                           << "s" << SIGMA
                           << "_PMEb" << static_cast<int>(100 * BETA)
                           << "g" << DELTA
                           << "_PreC" << CELL_WIDTH
                           << "sup" << SUPER_POINT_SIZE
                           << "_OERg" << static_cast<int>(100 * GAMMA_O)
                           << "s" << SIGMA_O
                           << "p" << static_cast<int>(100 * P_NOISE_O)
                           << "gr" << static_cast<int>(REMOVAL_ROADS_GROUPING_O)
                           << "mr" << static_cast<int>(REMOVAL_ROAD_MAXLENGTH_OPTION_O)
                           << ".csv";

    string OUTPUT_FILENAME = output_filename_stream.str();
    cout << "OUTPUT_FILENAME: " << OUTPUT_FILENAME << endl;

    // Intialize the arrays dynamically
    updateGridInfo(FORMAT_FILENAME);

    // Storage arrays for input gdf
    geoms = new tuple<int, int>[MAX_ROWS];
    speeds = new float[MAX_ROWS];
    angles = new int[MAX_ROWS];
    n_points = new int[MAX_ROWS];
    // ture_to_be_matched = new bool[MAX_ROWS];
    road_ids = new tuple<long long, long long, short>[MAX_ROWS];
    distance_to_road = new float[MAX_ROWS];
    types = new string[MAX_ROWS];
    to_be_matched = new bool[MAX_ROWS];
    new_road_ids = new tuple<long long, long long, short>[MAX_ROWS];
    new_distance_to_road = new float[MAX_ROWS];

    new_r_p_sim = new float[MAX_ROWS];
    new_road_angle = new int[MAX_ROWS];

    W_c = new double[FEATURE_MAX_ROWS]();
    W_s = new double[FEATURE_MAX_ROWS]();
    W_ms = new double[FEATURE_MAX_ROWS]();
    W_as = new double[FEATURE_MAX_ROWS]();
    W_ss = new double[FEATURE_MAX_ROWS]();
    W_r = new double[FEATURE_MAX_ROWS];
    W_Dir = new double[FEATURE_MAX_ROWS];

    N_c = new float[FEATURE_MAX_ROWS]();
    N_s = new float[FEATURE_MAX_ROWS]();
    N_ms = new float[FEATURE_MAX_ROWS]();
    N_as = new float[FEATURE_MAX_ROWS]();
    N_ss = new float[FEATURE_MAX_ROWS]();
    N_r = new int[FEATURE_MAX_ROWS];
    N_Dir = new int[FEATURE_MAX_ROWS];

    // 2D Arrays
    Cell2PointIndex = new int **[N_CELLS_X];
    Cell2PointsSize = new int *[N_CELLS_X];

    for (int i = 0; i < N_CELLS_X; i++)
    {
        Cell2PointIndex[i] = new int *[N_CELLS_Y];
        Cell2PointsSize[i] = new int[N_CELLS_Y];
        for (int j = 0; j < N_CELLS_Y; j++)
        {
            Cell2PointIndex[i][j] = nullptr; // no array allocated yet
            Cell2PointsSize[i][j] = 0;       // initialize sizes to 0
        }
    }

    int x_index,
        y_index,
        x_upper_lim, x_lower_lim,
        y_upper_lim, y_lower_lim,
        y_lower_start,
        Npoints_list_start,
        n_points_in_cell, n_points_in_Ncell,
        p_angle, q_angle,
        a_diff, AngleDiff,
        N_p, N_q,
        p, q;
    tuple<int, int> Ncell;
    float p_q_distance;
    double Weight, Adj_Weight;
    char ignore;
    auto *maxR = new unordered_map<tuple<long long, long long, short>, double, TupleHash3>[FEATURE_MAX_ROWS];
    auto *maxA = new unordered_map<int, double>[FEATURE_MAX_ROWS];

    cout << "Reading GDF" << endl;


    {
        ScopeProfiler profiler("read_gdf");
        int r = read_gdf(GDF_FILENAME);
        if (r == 1)
            return r;
    }
    // return 0;

    cout << "Loading JSON" << endl;
    unordered_map<tuple<int, int>, vector<int>, TupleHash> cell_to_points = load_json(INDEX_FILENAME);
    int *nearby_cell_points;

    cout << "Converting Index to 2D Array" << endl;
    vector<tuple<int, int>> cells_in_order;
    for (auto kv : cell_to_points)
    {
        auto [a, b] = kv.first;
        cells_in_order.push_back(kv.first);
        int size = kv.second.size();
        Cell2PointsSize[a][b] = size;
        Cell2PointIndex[a][b] = new int[size];
        for (int i = 0; i < size; ++i)
            Cell2PointIndex[a][b][i] = kv.second[i];
    }
    // TODO: Free the memory allocated for cell_to_points
    cell_to_points.clear();

    cout << "Computing MaxMatchScore and MaxSpeed" << endl;
    float MaxMatchScore = *max_element(new_distance_to_road, new_distance_to_road + MAX_ROWS);
    float MaxSpeed = *max_element(speeds, speeds + MAX_ROWS);
    sort(cells_in_order.begin(), cells_in_order.end(), compare_cell);

    // Calculate the raduis in cells
    cout << "Intializing Internal Variables" << endl;
    int radius_in_cells_x = 0;
    int radius_in_cells_y = 0;

    // Check if radius is smaller than half of cell width in x-direction
    if (D <= CELL_WIDTH_X / 2)
    {
        radius_in_cells_x = 0;
        cerr << "Warning: Radius is very small: " << D
             << ". Smaller than half the cell width in x: " << CELL_WIDTH_X / 2 << endl;
    }
    else
        radius_in_cells_x = ceil(D / CELL_WIDTH_X - 0.5);

    // Check if radius is smaller than half of cell width in y-direction
    if (D <= CELL_WIDTH_Y / 2)
    {
        radius_in_cells_y = 0;
        cerr << "Warning: Radius is very small: " << D
             << ". Smaller than half the cell width in y: " << CELL_WIDTH_Y / 2 << endl;
    }
    else
        radius_in_cells_y = ceil(D / CELL_WIDTH_Y - 0.5);

    int upper_i = 0;
    int n_pairs = 0;

    vector<Duration>
        d_o(2, Duration(0)),
        d_c(2, Duration(0)),
        d_i(2, Duration(0)),
        d_xy(2, Duration(0)),
        d_j(7, Duration(0));

    auto o0 = Clock::now();
    auto start_time = steady_clock::now();

    // for (int x_index = 0; x_index < N_CELLS_X; x_index++)
    // {
    //     for (int y_index = 0; y_index < N_CELLS_Y; y_index++)
    //     {
    //         if (Cell2PointIndex[x_index][y_index] == nullptr)
    //             continue;
    //         cout << "Cell (" << y_index << ", " << y_index << ") has " << Cell2PointsSize[y_index][y_index] << " points.\n";
    //     }
    // }
    // for (int i = 0; i < FEATURE_MAX_ROWS; ++i)
    //     maxR[i].clear(); //  = unordered_map<tuple<long long, long long, short>, double, TupleHash3>();

    {
        ScopeProfiler profiler("My expensive feature extraction");
        auto maxR_memBefore = approxUnorderedMapArrayMemory(maxR, FEATURE_MAX_ROWS);
        auto maxA_memBefore = approxUnorderedMapArrayMemory(maxA, FEATURE_MAX_ROWS);

        cout << "Starting Feature Extraction" << endl;
        // size_t memBefore = getMemoryUsageKB();
        // double cpuBefore = getCPUTime();
        for (const auto cell : cells_in_order)
        {
            update_progress(upper_i, cells_in_order.size(), 50, start_time);
            upper_i++;

            auto c0 = Clock::now();
            // Get the Cell Index
            x_index = get<0>(cell);
            y_index = get<1>(cell);

            // Calculate upper and lower limits for x and y indices // Found error
            x_lower_lim = x_index;
            x_upper_lim = min(x_index + radius_in_cells_x, N_CELLS_X - 1);

            y_lower_lim = max(0, y_index - radius_in_cells_y);
            y_upper_lim = min(y_index + radius_in_cells_y, N_CELLS_Y - 1);

            n_points_in_cell = Cell2PointsSize[x_index][y_index];

            auto c1 = Clock::now();
            d_c[0] += (c1 - c0);

            // Loop over Points of the Cell
            for (int i = 0; i < n_points_in_cell; i++)
            {
                auto i0 = Clock::now();
                p = Cell2PointIndex[x_index][y_index][i];
                if (p >= MAX_ROWS)
                {
                    cerr << "Error: Point index out of bounds: " << p << endl;
                    return 1;
                }
                N_p = n_points[p];

                auto i1 = Clock::now();
                d_i[0] += (i1 - i0);

                // Loop over the X-Cells of Nearby Cells
                for (int x = x_lower_lim; x <= x_upper_lim; x++)
                {
                    // First level
                    if (x == x_lower_lim) // only here
                        y_lower_start = y_index;
                    else
                        y_lower_start = y_lower_lim;

                    // SECOND Loop over the cells
                    for (int y = y_lower_start; y <= y_upper_lim; y++)
                    {
                        auto xy0 = Clock::now();

                        // Check if the cell exists
                        if (Cell2PointIndex[x][y] == nullptr)
                            continue;
                        else
                            nearby_cell_points = Cell2PointIndex[x][y]; // Get the points in the cell
                        n_points_in_Ncell = Cell2PointsSize[x][y];

                        if (x == x_index && y == y_index)
                            Npoints_list_start = i + 1; // Not to be repeated
                        else
                            Npoints_list_start = 0;

                        auto xy1 = Clock::now();
                        d_xy[0] += (xy1 - xy0);
                        for (int j = Npoints_list_start; j < n_points_in_Ncell; j++)
                        {
                            auto j0 = Clock::now();
                            n_pairs++;

                            q = nearby_cell_points[j];

                            if (q >= MAX_ROWS)
                            {
                                cerr << "Error: Point (Q) index out of bounds: " << q << endl;
                                return 1;
                            }
                            auto j1 = Clock::now();
                            N_q = n_points[q];
                            auto j2 = Clock::now();

                            p_q_distance = calc_distance(geoms[p], geoms[q]);
                            auto j3 = Clock::now();

                            if (p_q_distance > D)
                                continue;
                            auto j4 = Clock::now();

                            if (is_forbidden(types[p], types[q])) // TODO: Check this good
                                continue;
                            auto j5 = Clock::now();
                            double p_score, q_score, p_score_c, q_score_c;
                            p_score = new_distance_to_road[p];
                            q_score = new_distance_to_road[q];

                            p_angle = angles[p] / 90;
                            q_angle = angles[q] / 90;

                            Weight = pow(1 - p_q_distance / D, N); // Weight calculation

                            // Calculate AngleDiff
                            a_diff = abs(angles[p] - angles[q]);
                            AngleDiff = (a_diff <= 180) ? a_diff : a_diff - 180;

                            auto j6 = Clock::now();
                            if (types[p] != "old" && !(types[p] == "uncertain" && types[q] == "bad"))
                            {
                                // TODO: Check this as well
                                p_score_c = p_score;
                                q_score_c = q_score;
                                if (types[p] == "uncertain" && types[q] == "good")
                                    q_score_c = distance_to_road[q];
                                W_c[p] += N_q * Weight;
                                W_s[p] += N_q * Weight * q_score_c;
                                W_ms[p] += N_q * Weight * (1 - abs(p_score_c - q_score_c) / MaxMatchScore);
                                W_as[p] += N_q * Weight * (1 - AngleDiff / 180);
                                W_ss[p] += N_q * Weight * (1 - abs(speeds[p] - speeds[q]) / MaxSpeed);

                                double &r_w = maxR[p][road_ids[q]]; // Initialize it to zero if not existent
                                maxR[p][road_ids[q]] = max(r_w, Weight);
                                double &a_w = maxA[p][q_angle]; // Initialize it to zero if not existent
                                maxA[p][q_angle] = max(a_w, Weight);

                                // Adding the new features
                                N_c[p] += N_q;
                                N_s[p] += N_q * q_score_c;
                                N_ms[p] += N_q * (1 - abs(p_score_c - q_score_c) / MaxMatchScore);
                                N_as[p] += N_q * (1 - AngleDiff / 180);
                                N_ss[p] += N_q * (1 - abs(speeds[p] - speeds[q]) / MaxSpeed);
                            }

                            if (types[q] != "old" && !(types[q] == "uncertain" && types[p] == "bad"))
                            {
                                p_score_c = p_score;
                                q_score_c = q_score;
                                if (types[q] == "uncertain" && types[p] == "good")
                                    p_score_c = distance_to_road[p];
                                W_c[q] += N_p * Weight;
                                W_s[q] += N_p * Weight * p_score_c;
                                W_ms[q] += N_p * Weight * (1 - abs(p_score_c - q_score_c) / MaxMatchScore);
                                W_as[q] += N_p * Weight * (1 - AngleDiff / 180);
                                W_ss[q] += N_p * Weight * (1 - abs(speeds[p] - speeds[q]) / MaxSpeed);

                                double &r_w = maxR[q][road_ids[p]];
                                maxR[q][road_ids[p]] = max(r_w, Weight);
                                double &a_w = maxA[q][p_angle];
                                maxA[q][p_angle] = max(a_w, Weight);

                                N_c[q] += N_p;
                                N_s[q] += N_p * p_score_c;
                                N_ms[q] += N_p * (1 - abs(p_score_c - q_score_c) / MaxMatchScore);
                                N_as[q] += N_p * (1 - AngleDiff / 180);
                                N_ss[q] += N_p * (1 - abs(speeds[p] - speeds[q]) / MaxSpeed);
                            }
                            auto j7 = Clock::now();
                            d_j[0] += (j1 - j0);
                            d_j[1] += (j2 - j1);
                            d_j[2] += (j3 - j2);
                            d_j[3] += (j4 - j3);
                            d_j[4] += (j5 - j4);
                            d_j[5] += (j6 - j5);
                            d_j[6] += (j7 - j6);
                        }
                        auto xy2 = Clock::now();
                        d_xy[1] += (xy2 - xy1);
                    }
                }
                auto i2 = Clock::now();
                d_i[1] += (i2 - i1);
            }
            auto c2 = Clock::now();
            d_c[1] += (c2 - c1);
        }
        auto o1 = Clock::now();
        d_o[0] += (o1 - o0);

        cout << "Summing up maxR and maxA\n";

        for (int p = 0; p < FEATURE_MAX_ROWS; p++)
        {
            double sum = 0.00;
            for (const auto &pair : maxR[p])
                sum += pair.second;
            W_r[p] = sum;
            sum = 0.00;
            for (const auto &pair : maxA[p])
                sum += pair.second;
            W_Dir[p] = sum;

            N_r[p] = maxR[p].size();
            N_Dir[p] = maxA[p].size();
        }

        // for (int i = 0; i < N_CELLS_X; i++)
        // {
        //     for (int j = 0; j < N_CELLS_Y; j++)
        //     {
        //         if (Cell2PointIndex[i][j] != nullptr)
        //             continue;
        //         delete[] Cell2PointIndex[i][j];  // Deallocate each dynamic array
        //         Cell2PointIndex[i][j] = nullptr; // Optional: nullify the pointer for safety
        //     }
        // }
        nearby_cell_points = nullptr;
        auto o2 = Clock::now();
        d_o[1] += (o2 - o1);

        // size_t memAfter = getMemoryUsageKB();
        // double cpuAfter = getCPUTime();

        // Print the duration of each section
        cout << "Total time: " << duration_cast<seconds>(o2 - o0).count() << "s\n";
        cout << "Total pairs: " << n_pairs << endl;
        cout << "Total cells: " << cells_in_order.size() << endl;
        cout << "Total points: " << MAX_ROWS << endl;

        auto maxR_memAfter = approxUnorderedMapArrayMemory(maxR, FEATURE_MAX_ROWS);
        auto maxA_memAfter = approxUnorderedMapArrayMemory(maxA, FEATURE_MAX_ROWS);

        std::cout << "maxR Before: " << formatMemory(maxR_memBefore) << " and after: " << formatMemory(maxR_memAfter) << " bytes\n";
        std::cout << "maxA Before: " << formatMemory(maxA_memBefore) << " and after: " << formatMemory(maxA_memAfter) << " bytes\n";
        std::cout << "Estimated memory increase in maxR: " << formatMemory(maxR_memAfter - maxR_memBefore) << "\n";
        std::cout << "Estimated memory increase in maxA: " << formatMemory(maxA_memAfter - maxA_memBefore) << "\n";
    }

    print_duration_vector(d_o, "d_o");
    print_duration_vector(d_c, "d_c");
    print_duration_vector(d_i, "d_i");
    print_duration_vector(d_xy, "d_xy");
    print_duration_vector(d_j, "d_j");

    // Number of rows (assumes all arrays have the same size)
    ofstream file(OUTPUT_FILENAME);
    if (!file.is_open())
    {
        cerr << "Failed to open file.\n";
        return 1;
    }

    // Write header
    file << "lon,lat,speed,angle,matching_score,matched_road_src,matched_road_dest,matched_road_angle,r_p_sim,"
         << "W_c,W_s,W_ms,W_as,W_ss,W_r,W_Dir,"
         << "N_c,N_s,N_ms,N_as,N_ss,N_r,N_Dir,"
         << "type,N\n";

    // Write rows
    for (int p = 0; p < FEATURE_MAX_ROWS; p++)
    {
        file << get<0>(geoms[p]) << ","
             << get<1>(geoms[p]) << ","
             << speeds[p] << ","
             << angles[p] << ","
             << new_distance_to_road[p] << ","
             << get<0>(new_road_ids[p]) << ","
             << get<1>(new_road_ids[p]) << ","
             << new_road_angle[p] << ","
             << new_r_p_sim[p] << ","
             << W_c[p] << ","
             << W_s[p] << ","
             << W_ms[p] << ","
             << W_as[p] << ","
             << W_ss[p] << ","
             << W_r[p] << ","
             << W_Dir[p] << ","
             << N_c[p] << ","
             << N_s[p] << ","
             << N_ms[p] << ","
             << N_as[p] << ","
             << N_ss[p] << ","
             << N_r[p] << ","
             << N_Dir[p] << ","
             << types[p] << ","
             << n_points[p] << "\n";
    }

    cout << "Deleting Dynamic Arrays\n";
    delete[] maxR;
    delete[] maxA;

    delete[] geoms;
    delete[] speeds;
    delete[] angles;
    delete[] n_points;
    // delete[] ture_to_be_matched;
    delete[] road_ids;
    delete[] distance_to_road;
    delete[] types;
    delete[] to_be_matched;
    delete[] new_road_ids;
    delete[] new_distance_to_road;

    delete[] new_r_p_sim;
    delete[] new_road_angle;

    delete[] W_c;
    delete[] W_s;
    delete[] W_ms;
    delete[] W_as;
    delete[] W_ss;
    delete[] W_r;
    delete[] W_Dir;

    delete[] N_c;
    delete[] N_s;
    delete[] N_ms;
    delete[] N_as;
    delete[] N_ss;
    delete[] N_r;
    delete[] N_Dir;

    // Free 2D arrays
    for (int i = 0; i < N_CELLS_X; ++i)
    {
        for (int j = 0; j < N_CELLS_Y; ++j)
        {
            delete[] Cell2PointIndex[i][j]; // if allocated
        }
        delete[] Cell2PointIndex[i];
        delete[] Cell2PointsSize[i];
    }
    delete[] Cell2PointIndex;
    delete[] Cell2PointsSize;

    file.close();
    cout << "CSV file written successfully.\n";

    ostringstream output_time_filename_stream;
    output_time_filename_stream << "city_" << CITY
                                << "_pro/featureExTime_d" << D
                                << "n" << N
                                << "_SERg" << static_cast<int>(100 * GAMMA)
                                << "s" << SIGMA
                                << "_PMEb" << static_cast<int>(100 * BETA)
                                << "g" << DELTA
                                << "_PreC" << CELL_WIDTH
                                << "sup" << SUPER_POINT_SIZE
                                << "_OERg" << static_cast<int>(100 * GAMMA_O)
                                << "s" << SIGMA_O
                                << "p" << static_cast<int>(100 * P_NOISE_O)
                                << "gr" << static_cast<int>(REMOVAL_ROADS_GROUPING_O)
                                << "mr" << static_cast<int>(REMOVAL_ROAD_MAXLENGTH_OPTION_O)
                                << ".txt";

    ofstream time_file(output_time_filename_stream.str()); // use `std::ios::app` to append
    if (time_file.is_open())
    {
        time_file << fixed << setprecision(6)
                  << d_o[0].count();
        time_file.close();
    }
    else
        cerr << "Failed to open file for writing.\n";

    return 0;
}
