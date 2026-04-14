#include<bits/stdc++.h>
using namespace std;
const int MAXM = 55;      // 粒子群的大小
const int T = 60;         // 最大迭代次数
const int ARCHIVE_SIZE =30;  //外部存档最大容量
const int N_GRIDS = 10;   // 每个维度网格划分数
int cnt=0;
double w_max=0.9,w_min=0.4,w=w_max,c1=1.6,c2=1.8;
double r1,r2;

struct ty {
    double x1,x2;        // 位置
    double v;             // 速度（简化版，实际两个维度应有各自速度）
    double v1, v2;        // 补充：x1和x2方向的速度
    double p1_best, p2_best;  // 个人最优位置
    double pbest_f1, pbest_f2; // 个人最优的目标值
} gen[MAXM];

// 外部存档：存储帕累托前沿解 (f1, f2, x1, x2)
struct ArchiveNode {
    double f1, f2;
    double x1, x2;
};
vector<ArchiveNode> g_best;

// 随机数生成器
random_device rd;
mt19937 ge(rd());
uniform_real_distribution<double> dis1(-5.0, 5.0);   // 位置范围
uniform_real_distribution<double> dis2(0.0, 1.0);    // r1,r2 和速度初始值
uniform_real_distribution<double> disV(-1.0, 1.0);   // 速度范围

// 目标函数
double fitness1(double a, double b) {
    return a*a+b*b;
}

double fitness2(double a, double b) {
    return (a-2)*(a-2)+(b-2)*(b-2);
}

// 帕累托支配判断：a 是否支配 b
bool dominates(pair<double,double>a,pair<double,double>b) {
    // a支配b：a的所有目标 <= b，且至少有一个 <
    return (a.first<=b.first&&a.second<=b.second)&&
           (a.first < b.first||a.second<b.second);
}

// 更新惯性权重
double update_w(int t) {
    return w_max-(w_max-w_min)*1.0*t/T;
}

// 获取网格索引（用于网格划分法）
pair<int, int>get_grid_index(double f1,double f2,double min_f1,double max_f1,double min_f2,double max_f2) {
    int idx1=min(N_GRIDS-1,(int)((f1-min_f1)/(max_f1-min_f1+1e-10)*N_GRIDS));
    int idx2=min(N_GRIDS-1,(int)((f2-min_f2)/(max_f2-min_f2+1e-10)*N_GRIDS));
    return {idx1, idx2};
}

// 网格划分法维护存档
void prune_archive_by_grid() {
    if ((int)g_best.size()<=ARCHIVE_SIZE) return;
    // 计算当前存档的目标值范围
    double min_f1=g_best[0].f1,max_f1=g_best[0].f1;
    double min_f2=g_best[0].f2,max_f2=g_best[0].f2;
    for (auto &node:g_best) {
        min_f1=min(min_f1,node.f1);
        max_f1=max(max_f1,node.f1);
        min_f2=min(min_f2,node.f2);
        max_f2=max(max_f2,node.f2);
    }
    // 防止除零
    if (max_f1==min_f1) max_f1=min_f1+1e-6;
    if (max_f2==min_f2) max_f2=min_f2+1e-6;
    // 统计每个网格的解数量
    vector<vector<int>> grid_count(N_GRIDS,vector<int>(N_GRIDS,0)); //统计每个小格子里有多少个解
    vector<pair<int,int>> node_grid(g_best.size()); //记录每个解落在哪个格子里
    for (int i=0;i<(int)g_best.size();i++) {
        auto [gx, gy]=get_grid_index(g_best[i].f1,g_best[i].f2,min_f1,max_f1,min_f2,max_f2);
        node_grid[i]={gx, gy};
        grid_count[gx][gy]++;
    }
    // 找到最拥挤的网格
    int max_cnt=0;
    pair<int,int> max_grid = {0,0};
    for (int i=0;i<N_GRIDS;i++) {
        for (int j=0;j<N_GRIDS;j++) {
            if (grid_count[i][j]>max_cnt) {
                max_cnt=grid_count[i][j];
                max_grid={i, j};
            }
        }
    }
    // 从最拥挤的网格中随机删除一个解
    vector<int> to_remove;
    for (int i=0;i<(int)g_best.size();i++) {
        if (node_grid[i]==max_grid) {
            to_remove.push_back(i);
        }
    }
    if (!to_remove.empty()) {
        int idx=to_remove[rand()%to_remove.size()];
        g_best.erase(g_best.begin() + idx);
    }
}

// 更新外部存档
void update_archive(double f1,double f2,double x1,double x2){
    ArchiveNode newNode={f1,f2,x1,x2};
    // 检查新解是否被存档中任何解支配
    for (int i=0;i<(int)g_best.size();i++) {
        if (dominates({g_best[i].f1,g_best[i].f2},{f1, f2})) {
            return;  // 新解被支配，不加入
        }
    }
    // 删除所有被新解支配的存档解
    for (int i=(int)g_best.size()-1;i>=0;i--) {
        if (dominates({f1,f2},{g_best[i].f1,g_best[i].f2})) {
            g_best.erase(g_best.begin()+i);
        }
    }
    // 加入新解
    g_best.push_back(newNode);
    // 如果存档超限，用网格法修剪
    prune_archive_by_grid();
}

// 从存档中随机选择一个作为全局引导
ArchiveNode select_leader() {
    if (g_best.empty()) {
        // 如果存档为空，返回原点
        return {0,0,0,0};
    }
    int idx=rand()%g_best.size();
    return g_best[idx];
}

// 初始化
void init(){
    for (int i=0;i<MAXM;i++) {  // 改为从0开始
        gen[i].x1=dis1(ge);
        gen[i].x2=dis1(ge);
        gen[i].v1=disV(ge);  // 初始化x1方向速度
        gen[i].v2=disV(ge);  // 初始化x2方向速度
        double f1=fitness1(gen[i].x1,gen[i].x2);
        double f2=fitness2(gen[i].x1,gen[i].x2);

        // 个人最优初始化
        gen[i].p1_best=gen[i].x1;
        gen[i].p2_best=gen[i].x2;
        gen[i].pbest_f1=f1;
        gen[i].pbest_f2=f2;

        // 更新外部存档
        update_archive(f1,f2,gen[i].x1,gen[i].x2);
    }
}

// 更新粒子速度和位置
void update_gen(){
    cnt++;  // 迭代计数
    w=update_w(cnt);
    for (int i=0;i<MAXM;i++) {
        // 随机系数
        r1=dis2(ge);
        r2=dis2(ge);
        // 从存档中选择全局引导
        ArchiveNode leader=select_leader();
        // 速度更新公式
        gen[i].v1=w*gen[i].v1+c1*r1*(gen[i].p1_best-gen[i].x1)+c2*r2*(leader.x1-gen[i].x1); //leader.x1从存档中随机选一个
        gen[i].v2=w*gen[i].v2+c1*r1*(gen[i].p2_best-gen[i].x2)+c2*r2*(leader.x2-gen[i].x2); //leader.x2从存档中随机选一个

        // 位置更新
        gen[i].x1+=gen[i].v1;
        gen[i].x2+=gen[i].v2;

        // 边界处理（限制在[-5,5]范围内）
        gen[i].x1=max(-5.0,min(5.0,gen[i].x1));
        gen[i].x2=max(-5.0,min(5.0,gen[i].x2));

        // 速度限制（可选）
        gen[i].v1=max(-1.0,min(1.0,gen[i].v1));
        gen[i].v2=max(-1.0,min(1.0,gen[i].v2));

        // 计算新位置的目标值
        double new_f1=fitness1(gen[i].x1, gen[i].x2);
        double new_f2=fitness2(gen[i].x1,gen[i].x2);

        // 更新个人最优（帕累托支配规则）
        if (dominates({new_f1,new_f2},{gen[i].pbest_f1,gen[i].pbest_f2})) {
            // 新解支配旧pbest → 更新pbest
            gen[i].p1_best=gen[i].x1;
            gen[i].p2_best=gen[i].x2;
            gen[i].pbest_f1=new_f1;
            gen[i].pbest_f2=new_f2;
        } else if (!dominates({gen[i].pbest_f1, gen[i].pbest_f2}, {new_f1, new_f2})) {
            // 互不支配 → 随机决定是否更新（保持多样性）
            if (dis2(ge)<0.5) {
                gen[i].p1_best=gen[i].x1;
                gen[i].p2_best=gen[i].x2;
                gen[i].pbest_f1=new_f1;
                gen[i].pbest_f2=new_f2;
            }
        }
        // 如果旧pbest支配新解，则不变
        // 更新外部存档
        update_archive(new_f1,new_f2,gen[i].x1,gen[i].x2);
    }
}

// 输出结果
void print_result() {
    cout << "\n========== Final Pareto Front ==========\n";
    cout << "Total non-dominated solutions in archive: " << g_best.size() << "\n";
    cout << "    f1(x1,x2)        f2(x1,x2)        x1      x2\n";
    for (auto &node : g_best) {
        printf("  (%.4f, %.4f)  (%.4f, %.4f)  %.4f  %.4f\n",
               node.f1, node.f2, node.f1, node.f2, node.x1, node.x2);
    }

    // 输出理论前沿对比
    cout << "\nTheoretical Pareto front (x1 = x2, x in [0, 2]):\n";
    cout << "    f1              f2\n";
    for (double t = 0; t <= 2.1; t += 0.5) {
        double f1 = 2 * t * t;
        double f2 = 2 * (t - 2) * (t - 2);
        printf("  (%.4f, %.4f)\n", f1, f2);
    }
}

void solve() {
    srand(time(0));  // 随机种子
    init();
    for (int i = 1; i <= T; i++) {
        update_gen();
        if (i % 20 == 0) {
            cout << "Iteration " << i << ", archive size: " << g_best.size() << endl;
        }
    }
    print_result();
}
signed main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
/* 流程：
* 开始
  ↓
【1. 初始化】
  ├─ 随机生成粒子群的位置和速度
  ├─ 计算每个粒子的目标值(f1, f2)
  ├─ 设置个人最优(pbest) = 当前位置
  └─ 更新外部存档(帕累托前沿)
  ↓
【2. 主循环】(迭代T次)
  ↓
  ├─ 【2.1 选择引导者】
  │    └─ 从存档中随机选一个解作为全局引导(leader)
  ↓
  ├─ 【2.2 更新速度和位置】
  │    ├─ v = w*v + c1*r1*(pbest - x) + c2*r2*(leader - x)
  │    └─ x = x + v
  ↓
  ├─ 【2.3 边界处理】
  │    └─ 限制位置和速度在范围内
  ↓
  ├─ 【2.4 更新个人最优】
  │    ├─ 如果新解支配pbest → 更新pbest
  │    ├─ 如果互不支配 → 随机决定是否更新
  │    └─ 如果pbest支配新解 → 不更新
  ↓
  └─ 【2.5 更新外部存档】
       ├─ 检查新解是否被存档中任何解支配
       ├─ 删除所有被新解支配的解
       ├─ 加入新解
       └─ 如果存档超过容量 → 网格法修剪
  ↓
【3. 输出结果】
  └─ 打印帕累托前沿
  ↓
结束
 */