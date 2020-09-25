#include <bits/stdc++.h>
/**
*   输入输出
**/
inline bool scan_d(int &ret)
{
    char c; int sgn;
    if(c=getchar(),c==EOF) return 0; //EOF
    while(c!='−'&&(c<'0'||c>'9')) c=getchar();
    sgn=(c=='−')?−1:1;
    ret=(c=='−')?0:(c−'0');
    while(c=getchar(),c>='0'&&c<='9')
        ret=ret*10+(c−'0');
    ret*=sgn;
    return 1;
}

inline void out(int x)
{
    if(x>9) out(x/10);
    putchar(x%10+'0');
}

/**
*   莫队
**/
const int maxn = 1e5 + 7;
int arr[maxn];              //存原始数据
int visited[maxn];          //看题意。。有可能用到。。。
int ans[maxn];              //离线后按顺序存储答案
int num;                    //每次的答案

struct query{
    int l;
    int r;
    int id;
    query(){}
    query(int l, int r, int id) : l(l), r(r), id(id) {}
}qarr[maxn];                //存所有询问

int block;

bool cmp(const query & a, const query & b)
{
    if (a.l / block != b.l / block) return a.l < b.l;
    else    return a.r > b.r;
}

void add(int pos)
{
    //根据题意写
}

void del(int pos)
{
    //根据题意写
}

for (int i = 1; i <= q; i++)        //main中的操作
{
    cin >> qarr[i].l >> qarr[i].r;
    qarr[i].id = i;
}
sort(qarr + 1, qarr + m + 1, cmp);
int l = qarr[1].l;
int r = l - 1;
for (int i = 1; i <= m; i++)
{
    while (l < qarr[i].l)  del(l++);
    while (l > qarr[i].l)  add(--l);
    while (r < qarr[i].r)  add(++r);
    while (r > qarr[i].r)  del(r--);
    // 记录需要的数据，可能是ans数组，也可能是中间数组。。。
}
for (int i = 1; i <= q; i++)        //输出结果
    cout << ans[i] << endl;

/**
*   LIS
**/
const int maxn = 1e5+7;
int n;
int a[maxn];
int dp[maxn];

int lis()
{
    memset(dp, 0, sizeof dp);
    int len = 1;maxn
    dp[0] = a[0];
    for (int i = 1; i < n; ++i)
    {
        int pos = lower_bound(dp, dp + len, a[i]) - dp;
        dp[pos] = a[i];
        len = max(len, pos + 1);
    }
    return len;
}

/**
*   SPFA
**/
    /// 正常最短路
    void spfa(int i)
    {
        memset(visited, 0, sizeof(visited));
        for (int j = 0; j <= n ; j++)
            dis[j] = INT_MAX;
        dis[i] = 0;
        queue<int> q;
        int tmp;
        q.push(i);
		vis[i] = 1;
        while (!q.empty())
        {
            tmp = q.front();
            visited[tmp] = 0;
            q.pop();
            for (int j = 1; j <= n; j++)
                if (arr[tmp][j])
                {
                    if (dis[j] > dis[tmp] + arr[tmp][j])
                    {
                        dis[j] = dis[tmp] + arr[tmp][j];
                        if (!visited[j])
                            q.push(j), visited[j] = 1;
                    }
                }
        }
    }
    /// 最长路最短，改相应的if即可
    if (dis[j] > max(dis[tmp], arr[tmp][j]))
    {
        dis[j] = max(dis[tmp], arr[tmp][j]);
        if (!visited[j])
            q.push(j), visited[j] = 1;
    }
    /// 最短路最长，改相应的if以及dis初始化相反即可
    if (dis[j] < min(dis[tmp], arr[tmp][j]))
    {
        dis[j] = min(dis[tmp], arr[tmp][j]);
        if (!visited[j])
            q.push(j), visited[j] = 1;
    }
    /// 判负环回路，记得开头memset
    int cnt[MAXN];          //每个点的入队列次数
    memset(cnt, 0, sizeof cnt);
    cnt[start] = 1;
    if (!visited[j])
    {
        q.push(j), visited[j] = 1;
        if(++cnt[v] > n)return false;
    }

	/// 邻接表版(实用版)

	const int maxn = 3e4+7;
	const int maxm = 2e5+7;

	struct Edge{
		int to;
		int v;
		int next;
	}edge[maxm];
	int tol;
	// vector<pair<int, int> > v[maxn];		要求时间不紧可以用vector
	int vis[maxn], dis[maxn], head[maxn];
	int n, m;

	void add(int a, int b, int v)
	{
		edge[tol].to = b;
		edge[tol].v = v;
		edge[tol].next = head[a];
		head[a] = tol++;
	}

	void spfa()
	{
		for (int i = 1; i <= n; i++)
			dis[i] = INT_MAX / 2;
		stack<int> s;
		s.push(1);
		vis[1] = 1;
		dis[1] = 0;
		while (!s.empty())
		{
			int tmp = s.top();
			s.pop();
			vis[tmp] = 0;
			for (int i = head[tmp]; i != -1; i = edge[i].next)
				if (dis[edge[i].to] > dis[tmp] + edge[i].v)
				{
					dis[edge[i].to] = dis[tmp] + edge[i].v;
					if (!vis[edge[i].to])
						s.push(edge[i].to), vis[edge[i].to] = 1;
				}
		}
	}

	int main()
	{
		memset(head, -1, sizeof head);
		scanf("%d%d", &n, &m);
		int f, t, w;
		for (int i = 0; i < m; i++)
		{
			scanf("%d%d%d", &f, &t, &w);
			add(f, t, w);
		}
		spfa();
		printf("%d\n", dis[n]);
		return 0;
	}



/**
*   并查集
**/
int ufs[maxn];    //并查集

void init(int n)    //初始化
{
    for (int i = 0; i < n; i++)
        ufs[i] = i;
}

int getRoot(int a)    //获得a的根节点。路径压缩
{
    if (ufs[a] != a)    //没找到根节点
        ufs[a] = GetRoot(ufs[a]);
	return ufs[a];
}

void Merge(int a, int b)    //合并a和b的集合
{
    ufs[GetRoot(b)] = GetRoot(a);
}

bool query(int a, int b)    //查询a和b是否在同一集合
{
    return GetRoot(a) == GetRoot(b);
}

int sum = 0;
for(int i = 1; i <= n; i++)     //统计连通分量个数
{
    if(ufs[i] == i)
        sum++;
}

/**
*   MST(Kruskal)
**/
const int maxn = 110;//最大点数
const int maxm = 10000;//最大边数
int F[maxn];//并查集使用

struct Edge{
    int u, v, w;
}edge[maxm];//存储边的信息，包括起点/终点/权值

int tol;//边数，加边前赋值为0

void addedge(int u, int v, int w)
{
    edge[tol].u = u;
    edge[tol].v = v;
    edge[tol++].w = w;
}

//排序函数，讲边按照权值从小到大排序
bool cmp(const Edge & a, const Edge & b)
{
    return a.w < b.w;
}
int Find(int x)
{
    if(F[x] == -1)    return x;
    return F[x] = Find(F[x]);
}
//传入点数，返回最小生成树的权值，如果不连通返回-1
int Kruskal(int n)
{
    memset(F, -1, sizeof F);
    sort(edge, edge + tol, cmp);
    int cnt = 0;
    int ans = 0;
    for (int i = 0; i < tol; i++)
    {
        int u = edge[i].u;
        int v = edge[i].v;
        int w = edge[i].w;
        int t1 = Find(u);
        int t2 = Find(v);
        if (t1 != t2)
        {
            ans += w;
            F[t1] = t2;
            cnt++;
        }
        if (cnt == n-1)  break;
    }
    if (cnt < n-1) return -1;
    else return ans;
}

/**
*   分组背包
**/
for (int i = 1; i <= n; i++)
    for (int j = m; j >= 1; j--)
        for (int k = 1; k <= j; k++)
            dp[j] = max(dp[j], dp[j - k] + arr[i][k]);


/**
*   矩阵快速幂
**/
struct matrix{
	long long mat[3][3];                    //n 可变

	matrix() { memset(mat, 0, sizeof mat); }

	matrix operator * (const matrix &P)
	{
		matrix ans;
		for(int k=0;k<3;k++)
			for(int i=0;i<3;i++)
				for(int j=0;j<3;j++)
					ans.mat[i][j]=(ans.mat[i][j]+mat[i][k]*P.mat[k][j]%mod)%mod;
		return ans;
	}
}unit, COE, ans;            //unit 单位矩阵    COE 快速幂矩阵    ans 答案矩阵

matrix mat_quick_pow(matrix X,long long m)
{
	matrix ans;
	for(ans=unit;m;m>>=1,X=X*X)
		if(m&1)
			ans=ans*X;
	return ans;
}
void init(long long A,long long B,long long C,long long D,long long x)
{                                   //初始化根据题目写
	COE.mat[0][0]=0;
	COE.mat[0][1]=1;
	COE.mat[0][2]=0;
	COE.mat[1][0]=C;
	COE.mat[1][1]=D;
	COE.mat[1][2]=x;                //x 可变，计算带常数的才会用到
	COE.mat[2][0]=0;
	COE.mat[2][1]=0;
	COE.mat[2][2]=1;
	return;
}

/**
*   普通快速幂
**/

long long quick_pow(long long a, long long b, int mod)
{
    long long res = 1;
    while (b > 0)
    {
        a = a % mod;
        if (b & 1)
            res = res * a % mod;
        b = b >> 1;
        a = a * a % mod;
    }
    return res;
}

/**
*   最快速求mod(1e9+9)的斐波那契数(只能 mod 1e9+9)
**/

#include <bits/stdc++.h>

using namespace std;

const int mod = 1e9+9;

long long quick_pow(long long a, long long b)
{
    long long res=1;
    while(b>0)
    {
        a=a%mod;
        if(b&1)
            res=res*a%mod;
        b=b>>1;
        a=a*a%mod;
    }
    return res;
}

void getFab(long long n)    //好像是二次剩余得到的值
{
    return 276601605 * ((quick_pow(691504013, n) - quick_pow(308495997, n) + mod) % mod) % mod;
}


/**
*   组合数
**/
typedef long long ll;
const ll maxn = 400500, mod = 998244353;
ll fact[maxn],f[maxn],inv[maxn],cal[maxn];

void init()
{
    fact[0]=f[0]=inv[1]=inv[0]=1;
    for(ll i=2;i<maxn;++i)
        inv[i]=inv[mod%i]*(mod-mod/i)%mod;
    for(ll i=1;i<maxn;++i)
    {
        fact[i]=fact[i-1]*i%mod;
        f[i]=f[i-1]*inv[i]%mod;
    }
}

ll C(ll n,ll m)
{
    if(n<m||n<0||m<0)
        return 0;
    return (fact[n]*f[m]%mod*f[n-m]%mod)%mod;
}

/**
*   扩展GCD
**/
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

int extgcd(int a, int b, int &x, int &y)
{
    if (b == 0) { x = 1, y = 0; return a; }
    int d = extgcd(b, a % b, x, y);
    int t = x;
    x = y;
    y = t - a / b * y;
    return d;
} // a*x + b*y = d

/**
*   素数筛(线性)
**/

const int maxn = 1e5+7;
int prime[maxn];
int visited[maxn];
int cnt = 0;

void init()
{
	memset(visited, 0, sizeof(visited));
	for (int i = 2; i < maxn; ++i)
	{
	  if (!visited[i])
		prime[cnt++] = i;
	  for (int j = 0; j < cnt && prime[j]*i < maxn; ++j)
	  {
		visited[i*prime[j]] = 1;
		if (i % prime[j] == 0)
		  break;
	  }
	}
}

/**
*   质因数分解
**/
map<lld,int> fact;
const size_t maxn = 1000005;
char isp[maxn];
void init();                // 素数筛（见上面）
void expand(lld n)
{
    fact.clear();
    for (lld i = 2; i*i <= n;)
    {
        if (n % i == 0)
        while (n % i == 0)
            fact[i]++, n /= i;
        while (isp[++i] == 0);
    }
    if (n > 1) fact[n]++;
}
/*  遍历map
    for (auto it = fact.begin(); it != fact.end(); ++it)
        cout << it->first << " " << it->second << endl;
*/

/**
*   欧拉函数	(一个数的所有质因子之和为phi[n]*n/2)
**/
const int MAXN = 10001;
unsigned phi[MAXN];
void init()                     //递推打表欧拉函数
{
    for (int i = 1; i <= MAXN; i++)
        phi[i] = i;
    for (int i = 2; i <= MAXN; i += 2)
        phi[i] /= 2;
    for (int i = 3; i <= MAXN; i += 2)
        if (phi[i] == i)
        {
            for (j = i; j <= MAXN; j += i)
            phi[j] = phi[j] / i * (i-1);
        }
}
long long phi(long long x)        //公式计算欧拉函数
{
    long long i, res = x;
    for (i = 2; i*i <= x+1; i++)
        if (x % i == 0)
        {
            res = res / i * (i-1);
            while (x % i == 0) x /= i;
        }
    if (x > 1)
        res = res / x * (x-1);
    return res;
}

/**
*   同余函数
**/
int modeq(int a, int b, int n, int* &r)
{                               // n > 0
    int e, i, d, x, y;
    d = extgcd(a, n, x, y);
    if (b % d)  return 0;
    else
    {
        r = new int[n/d];
        e = (x*(b/d)) % n;
        for (i = 0; i < d; i++)
            r[i] = (e + i*(n/d)) % n;
        return d;
    }
}   // 返回了同余方程解的个数

/**
*   线性递推出规律
**/

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i,a,n) for(int i=a;i<n;i++)
namespace linear
{
    ll mo=1000000009;
    vector<ll> v;
    double a[105][105],del;
    int k;
    struct matrix
    {
        int n;
        ll a[50][50];
        matrix operator * (const matrix & b)const
        {
            matrix c;
            c.n=n;
            rep(i,0,n)rep(j,0,n)c.a[i][j]=0;
            rep(i,0,n)rep(j,0,n)rep(k,0,n)
            c.a[i][j]=(c.a[i][j]+a[i][k]*b.a[k][j]%mo)%mo;
            return c;
        }
    }A;
    bool solve(int n)
    {
        rep(i,1,n+1)
        {
            int t=i;
            rep(j,i+1,n+1)if(fabs(a[j][i])>fabs(a[t][i]))t=j;
            if(fabs(del=a[t][i])<1e-6)return false;
            rep(j,i,n+2)swap(a[i][j],a[t][j]);
            rep(j,i,n+2)a[i][j]/=del;
            rep(t,1,n+1)if(t!=i)
            {
                del=a[t][i];
                rep(j,i,n+2)a[t][j]-=a[i][j]*del;
            }
        }
        return true;
    }
    void build(vector<ll> V)
    {
        v=V;
        int n=(v.size()-1)/2;
        k=n;
        while(1)
        {
            rep(i,0,k+1)
            {
                rep(j,0,k)a[i+1][j+1]=v[n-1+i-j];
                a[i+1][k+1]=1;
                a[i+1][k+2]=v[n+i];
            }
            if(solve(n+1))break;
            n--;k--;
        }
        A.n=k+1;
        rep(i,0,A.n)rep(j,0,A.n)A.a[i][j]=0;
        rep(i,0,A.n)A.a[i][0]=(int)round(a[i+1][A.n+1]);
        rep(i,0,A.n-2)A.a[i][i+1]=1;
        A.a[A.n-1][A.n-1]=1;
    }
    void formula()
    {
        printf("f(n) =");
        rep(i,0,A.n-1)printf(" (%lld)*f(n-%d) +",A.a[i][0],i+1);
        printf(" (%lld)\n",A.a[A.n-1][0]);
    }
    ll cal(ll n)
    {
        if(n<v.size())return v[n];
        n=n-k+1;
        matrix B,T=A;
        B.n=A.n;
        rep(i,0,B.n)rep(j,0,B.n)B.a[i][j]=i==j?1:0;
        while(n)
        {
            if(n&1)B=B*T;
            n>>=1;
            T=T*T;
        }
        ll ans=0;
        rep(i,0,B.n-1)ans=(ans+v[B.n-2-i]*B.a[i][0]%mo)%mo;
        ans=(ans+B.a[B.n-1][0])%mo;
        while(ans<0)ans+=mo;
        return ans;
    }
}

int main()
{
//  vector<ll> V={1 ,4 ,9 ,16,25,36,49};
//  vector<ll> V={1 ,1 ,2 ,3 ,5 ,8 ,13};
//  vector<ll> V={2 ,2 ,3 ,4 ,6 ,9 ,14};
    vector<ll> V={1,1,1,3,5,9,17};//<-----
    linear::build(V);
    linear::formula();
    ll n;
    while(~scanf("%lld",&n))
    {
        printf("%lld\n",linear::cal(n-1));
    }
    return 0;
}

/**
*   匈牙利算法
**/

//cx[i]表示与X部i点匹配的Y部顶点编号
//cy[i]表示与Y部i点匹配的X部顶点编号

bool dfs(int u){
    for(int v=1;v<=m;v++)
        if(t[u][v]&&!vis[v]){
            vis[v]=1;
            if(cy[v]==-1||dfs(cy[v])){
                cx[u]=v;cy[v]=u;
                return 1;
            }
        }
    return 0;
}
int maxmatch()//匈牙利算法主函数
{
    int ans=0;
    memset(cx,0xff,sizeof cx);
    memset(cy,0xff,sizeof cy);
    for(int i=0;i<=nx;i++)
        if(cx[i]==-1)//如果i未匹配
        {
            memset(visit,false,sizeof(visit)) ;
            ans += dfs(i);
        }
    return ans ;
}

/**
*   最大流
**/

/** Ford-Fulkerson (O(FE)) 较慢，容易被卡 **/
const int maxn = 205;
const int inf = INT_MAX / 2;

struct edge{
    int to;
    int cap;
    int rev;
};

vector<edge> G[maxn];
bool vis[maxn];
int arr[maxn][maxn];		//解决重边问题，多测记得memset

void add_edge(int from, int to, int cap)
{
    G[from].push_back((edge){to, cap, G[to].size()});
    G[to].push_back((edge){from, 0, G[from].size() - 1});
}

int dfs(int v, int t, int f)
{
    if (v == t) return f;
    vis[v] = true;
    for (int i = 0; i < G[v].size(); i++)
    {
        edge & e = G[v][i];
        if (!vis[e.to] && e.cap > 0)
        {
            int d = dfs(e.to, t, min(f, e.cap));
            if (d > 0)
            {
                e.cap -= d;
                G[e.to][e.rev].cap += d;
                return d;
            }
        }
    }
    return 0;
}

int max_flow(int s, int t)
{
    int flow = 0;
    while (1)
    {
        memset(vis, 0, sizeof vis);
        int f = dfs(s, t, INF);
        if (!f) return flow;
        flow += f;
    }
}

/** EK算法模板（s为源点，t为汇点，函数返回值为最大流）**/
//p储存路径，a用来计算路径
int EK(int s,int t)  
{  
    queue<int> q;  
    int p[maxn<<1],a[maxn<<1];  
    int f=0;  
    while(1)  
    {  
        memset(a,0,sizeof(a));  
        a[s]=INF;  
        q.push(s);  
        while(!q.empty())  
        {  
            int u=q.front();  
            q.pop();  
            for(int v=0; v<=t; ++v)  
                if(!a[v]&&arr[u][v])  
                {  
                    p[v]=u,q.push(v);  
                    a[v]=min(a[u],arr[u][v]);  
                    if(v==t) break;  
                }  
        }  
        if(a[t]==0) break;  
        for(int u=t; u!=s; u=p[u])  
        {  
            arr[p[u]][u]-=a[t];  
            arr[u][p[u]]+=a[t];  
        }  
        f+=a[t];  
    }  
    return f;  
}  

/** Dinic (O(EV*V) **/
const int maxn = 205;
const int INF = INT_MAX / 2;

struct edge{
    int to, cap, rev;
};

vector<edge> G[maxn];
int level[maxn];
int iter[maxn];

void add_edge(int from, int to, int cap)
{
    G[from].push_back((edge){to, cap, G[to].size()});
    G[to].push_back((edge){from, 0, G[from].size() - 1});
}

void bfs(int s)
{
	memset(level, -1, sizeof level);
	queue<int> q;
	level[s] = 0;
	q.push(s);
	while (!q.empty())
	{
		int v = q.front();
		q.pop();
		for (int i = 0; i < G[v].size(); i++)
		{
			edge & e = G[v][i];
			if (e.cap > 0 && level[e.to] < 0)
				level[e.to] = level[v] + 1, q.push(e.to);
		}
	}
}

int dfs(int v, int t, int f)
{
    if (v == t) return f;
    
    for (int & i = iter[v]; i < G[v].size(); i++)
    {
        edge & e = G[v][i];
		if (e.cap > 0 && level[v] < level[e.to])
		{
            int d = dfs(e.to, t, min(f, e.cap));
            if (d > 0)
            {
                e.cap -= d;
                G[e.to][e.rev].cap += d;
                return d;
            }
        }
    }
    return 0;
}

int max_flow(int s, int t)
{
	int flow = 0;
	for (;;)
	{
		bfs(s);
		if (level[t] < 0)	return flow;
		memset(iter, 0, sizeof iter);
		int f;
		while ((f = dfs(s, t, INF)) > 0)
			flow += f;
	}
}

/**
*	求第k大/小的数（快排改进）
**/
#include<stdio.h>
#include<string.h>

int a[100];

int par(int l,int r){
    int x=a[l];
    while(l<r){
        while(l<r&&a[r]<=x)    --r;
        a[l]=a[r];
        while(l<r&&a[l]>=x)    ++l;
        a[r]=a[l];
    }
    a[l]=x;
    return l;
}
int search(int l,int r,int k){
    if(l<=r){
        int p = par(l,r);
        if(p-l+1==k)    return p;
        else if(p-l+1<k){
            return search(p+1,r,k-(p-l+1));
        }else{
            return search(l,p-1,k);
        }
    }
}
int main(){
    int n,k;
    scanf("%d%d",&n,&k);
    for(int i=1;i<=n;i++)
        scanf("%d",&a[i]);
    printf("%d\n",a[search(1,n,k)]);
    return 0;
}

/**
*	斯特林数判断阶乘位数
**/
const double e = 2.71828182845;
const double pi = 3.1415926;

int main() 
{
	int t, i, f, v;
	double a, s;
	const double log10_e = log10(e);
	const double log10_2_pi = log10(2.0*pi)/2.0;
	while (scanf("%d", &t) != EOF && t) 
	{
		for (i = 0; i < t; ++i) 
		{
			scanf("%d\n", &v);
			if (1 == v) 
			{
				printf("1\n");
				continue;
			}
			a = v;
			s = log10_2_pi + (a+0.5)*log10(a) - a * log10_e;
			f = ceil(s);
			printf("%d\n", f); 
		}
	}
	return 0; 
}

/**
*   计算几何
**/

#include <bits/stdc++.h>

using namespace std;

const double eps = 1e-6;
const double PI = acos(-1.0);

typedef struct Point{
    double x, y;
    Point(){}
    Point(double x, double y) : x(x), y(y) {}
}Vector;

Vector operator + (Vector a, Vector b)
{
    return Vector(a.x + b.x, a.y + b.y);
}

Vector operator - (Vector a, Vector b)
{
    return Vector(a.x - b.x, a.y - b.y);
}

Vector operator * (Vector a, double p)
{
    return Vector(a.x * p, a.y * p);
}

Vector operator / (Vector a, double p)
{
    return Vector(a.x / p, a.y / p);
}

int dcmp(double x)
{
    if (fabs(x) < eps)  return 0;
    else if (x > 0)     return 1;
    return -1;
}

bool operator == (const Vector & a, const Vector & b)
{
    return dcmp(a.x - b.x) == 0 && dcmp(a.y - b.y) == 0;
}

double Dot(Vector a, Vector b)  // 内积
{
    return a.x * b.x + a.y * b.y;
}

double Length(Vector a) // 模
{
    return sqrt(Dot(a, a));
}

double Angle(Vector a, Vector b)    //夹角,弧度制
{
    return acos(Dot(a, b)/ Length(a) / Length(b));
}

double Cross(Vector a, Vector b)    //叉积
{
    return a.x * b.y - a.y * b.x;
}

Vector Rotate(Vector a, double rad) //逆时针旋转
{
    return Vector(a.x * cos(rad) - a.y * sin(rad),
                  a.x * sin(rad) + a.y * cos(rad));
}

double Distance(Point a, Point b)   //两点间距离
{
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2))
}

double Area(Point a, Point b, Point c)  //三角形面积
{
    return fabs(Cross(b - a, c - a) / 2);
}






















