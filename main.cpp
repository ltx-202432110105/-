#include<bits/stdc++.h>
using namespace std;
const int T=100;
const int Generation=50;
int t;
double g_best,c1=1.6,c2=1.8,r1,r2,w_max=0.9,w_min=0.4,w=w_max,g_best_post;
struct ty {
    double v;
    double x;
    double p;
}generation[Generation];
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis1(0.0,4.0);   // 位置范围
uniform_real_distribution<double> dis2(0.0,1.0);   // 随机数r1,r2
uniform_real_distribution<double> dis3(-0.5,0.5);  // 速度范围

double fitness(double x) {
    return 1-cos(3*x)*exp(-x);
}
double update_w(int time) {
    return w_max-(w_max-w_min)*1.0*time/T;
}
void init() {
    g_best=-100000000.0;
    for (int i=1;i<=Generation;i++) {
        generation[i].x=dis1(gen);   // 随机初始化位置
        generation[i].v=dis3(gen);   // 随机初始化速度
        double fit=fitness(generation[i].x);  // 计算适应度
        generation[i].p=generation[i].x;      // 个体最优位置
        if (fit>g_best) {
            g_best=fit;
            g_best_post=generation[i].x;
        }
    }
}
void update_generation() {
    t++;
    for (int i=1;i<=Generation;i++) {
        r1 = dis2(gen);
        r2 = dis2(gen);
        generation[i].v=w*generation[i].v+c1*r1*(generation[i].p-generation[i].x)+c2*r2*(g_best_post-generation[i].x);
        generation[i].x=generation[i].x+generation[i].v;
        //边界处理
        if (generation[i].x<0) generation[i].x=0;
        if (generation[i].x>4) generation[i].x=4;
        double current_fit=fitness(generation[i].x);
        double p_best_fit=fitness(generation[i].p);
        // 更新个体最优（比较适应度，存位置）
        if (current_fit>p_best_fit) {
            generation[i].p=generation[i].x;
        }
        // 更新全局最优
        if (current_fit>g_best) {
            g_best=current_fit;
            g_best_post=generation[i].x;
        }
        w=update_w(t);
    }
}
void solve() {
    init();
    for (int i=1;i<=T;i++) {
        update_generation();
    }
    cout << g_best << endl;
}

signed main(){
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cout << fixed << setprecision(15);
    int _=1;
    //cin>>_;
    while (_--) {
        solve();
    }
    return 0;
}