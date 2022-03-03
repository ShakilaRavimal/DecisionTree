#include <iostream>
#include <vector>
#include "bits-stdc++.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class node{

public:

    double Average;
    double MSE;
    double Threshold;
    vector<vector<double>> Tdatareg;
    string Attr;
    int colnum;
    pair<int,string> path;
    vector<vector<string>> Tdata;

};

const int SIZE = 2000;
string adj_rules[SIZE][SIZE];
vector<node> nodes;

string InttoString(double a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}

vector<string> split(const string& str,const string& delim)
{
    vector<string> tokens;
    size_t prev =0,pos=0;
    do
    {
        pos = str.find(delim,prev);
        if(pos==string::npos)
        {
            pos = str.length();
        }
        string token = str.substr(prev,pos-prev);
        if(!token.empty())
        {
            tokens.push_back(token);
        }
        prev = pos+delim.length();
    }
    while(pos<str.length() && prev<str.length());

    return tokens;
}

string& remove_charss(string& s,const string& chars)
{
    s.erase(remove_if(s.begin(),s.end(),[&chars](const char& c){

                      return chars.find(c)!=string::npos;

                      }),s.end());

    return s;
}

void add_rules(const int& start,const int& end,const string& rule)
{
    adj_rules[start][end] = rule;
}

void add_node(const string& attr,vector<vector<string>>& tdata,pair<int,string>& path,const double& threshold,const double& average,const double& mse,vector<vector<double>>& tdatareg)
{
    node* newnode = new node();
    newnode->Attr = attr;
    newnode->Tdata = tdata;
    newnode->colnum = NULL;
    newnode->path = path;
    newnode->Threshold = threshold;
    newnode->Average = average;
    newnode->MSE;
    newnode->Tdatareg = tdatareg;

    nodes.push_back(*newnode);
}

void drop_rows(vector<vector<string>>& tdata,const int& i)
{
    tdata[i].clear();
}

template<typename T>
void shuffle_rows(vector<vector<T>>& tdata,time_t seed)
{
    srand((unsigned)seed);
    vector<T> saved;
    for(int i=1;i<tdata.size();i++)
    {
        int r = rand()%tdata.size();
        if(r!=i && r!=0)
        {
            for(int j=0;j<tdata[i].size();j++)
            {
                saved.push_back(tdata[i][j]);
            }
            drop_rows(tdata,i);
            for(int j=0;j<saved.size();j++)
            {
                tdata[i].push_back(tdata[r][j]);
            }
            drop_rows(tdata,r);
            for(int j=0;j<saved.size();j++)
            {
                tdata[r].push_back(saved[j]);
            }
            saved.clear();

        }

    }
}

double GiniimpuritySub(const string& sub,const int& j,const int& index,set<string>& targets)
{
    vector<double> firstcountarr;

    double val = 1;
    for(auto& x : targets)
    {
        double firstcount = 0;
        for(int i=1;i<nodes[index].Tdata.size();i++)
        {
            if(nodes[index].Tdata[i].size()!=0)
            {
                if(nodes[index].Tdata[i][j]==sub)
                {
                    if(nodes[index].Tdata[i][nodes[index].Tdata[0].size()-1]==x)
                    {
                        firstcount++;
                    }

                }
            }


        }
        firstcountarr.push_back(firstcount);
    }

    double total = accumulate(firstcountarr.begin(),firstcountarr.end(),0.0);
    for(int i=0;i<firstcountarr.size();i++)
    {
        val -= (pow(firstcountarr[i]/(total),2));
    }

    return val;

}

unordered_map<string,double> count_Features(const int& j,const int& index,int& tcount)
{
    set<string> subfeature;
    vector<string> copyfeature;
    for(int i=1;i<nodes[index].Tdata.size();i++)
    {
        if(nodes[index].Tdata[i].size()!=0)
        {
            subfeature.insert(nodes[index].Tdata[i][j]);
            copyfeature.push_back(nodes[index].Tdata[i][j]);
        }

    }

    unordered_map<string,double> Umap;

    for(auto& x : subfeature)
    {
        Umap.insert(make_pair(x,count(copyfeature.begin(),copyfeature.end(),x)));

    }

    tcount = copyfeature.size();
    return Umap;
}

double cal_gini_impurity(const int& j,const int& index,set<string>& targets)
{
    int tcount;
    unordered_map<string,double> Umap = count_Features(j,index,tcount);

    double SumofGini = 0;
    for(auto& x : Umap)
    {
        SumofGini += (GiniimpuritySub(x.first,j,index,targets)*(x.second/tcount));
    }

    return SumofGini;

}

template<typename T>
void column_drop(vector<int> drops,vector<vector<T>>& tdata)
{
    sort(drops.begin(),drops.end());
    for(int k=0;k<drops.size();k++)
    {
        if(k>0)
        {
            drops[k] = drops[k]-1;
        }
        for(int i=0;i<tdata.size();i++)
        {
            tdata[i].erase(tdata[i].begin()+drops[k]);
        }

    }
}

void header_col_Drop(vector<int> drops,vector<string>& header)
{
    sort(drops.begin(),drops.end());
    for(int k=0;k<drops.size();k++)
    {
        if(k>0)
        {
            drops[k] = drops[k]-1;
        }

        header.erase(header.begin()+drops[k]);

    }

}

string set_node(int& colnum,const int& index,set<string>& targets,const double& single_node_accuracy_per)
{
    vector<double> collected;
    double low = 10000;
    int feature_index = 0;
    for(int j=0;j<nodes[index].Tdata[0].size()-1;j++)
    {
        double Gini = cal_gini_impurity(j,index,targets);
        collected.push_back(Gini);
        if(low>Gini)
        {
            low = Gini;
            feature_index = j;
        }
    }

    collected.erase(unique(collected.begin(),collected.end()),collected.end());
    colnum = feature_index;
    if(collected.size()==1)
    {
        map<string,int> mp;
        for(int h=1;h<nodes[index].Tdata.size();h++)
        {
            mp[nodes[index].Tdata[h][nodes[index].Tdata[h].size()-1]]++;
        }
        int maxv = 0;
        string maxt = "";
        for(auto& x : mp)
        {
            if(maxv<x.second)
            {
                maxv = x.second;
                maxt = x.first;
            }
        }
        colnum = 10000;
        return maxt;
    }
    else
    {
        map<string,double> freqmp;
        for(int i=1;i<nodes[index].Tdata.size();i++)
        {
            freqmp[nodes[index].Tdata[i][nodes[index].Tdata[i].size()-1]]++;
        }

        for(auto& x : freqmp)
        {
            double per = (x.second/nodes[index].Tdata.size())*100;
            if(per>=single_node_accuracy_per)
            {
                colnum = 10000;
                return x.first;
            }
        }

        return nodes[index].Tdata[0][feature_index];

    }

}

unordered_set<string> Get_featurearr(const int& colnum,const int& origin)
{
    unordered_set<string> subfeature;
    for(int i=1;i<nodes[origin].Tdata.size();i++)
    {
        if(nodes[origin].Tdata[i].size()!=0)
        {
            subfeature.insert(nodes[origin].Tdata[i][colnum]);
        }

    }

    return subfeature;
}

vector<vector<string>> getTdata(pair<int,string>& path,const int& origin)
{
    vector<vector<string>> tdata;
    tdata.resize(nodes[origin].Tdata.size());
    int k=0;

    for(int j=0;j<nodes[origin].Tdata[0].size();j++)
    {
        tdata[k].push_back(nodes[origin].Tdata[0][j]);

    }

    k++;

    for(int i=0;i<nodes[origin].Tdata.size();i++)
    {
        if(nodes[origin].Tdata[i].size()!=0)
        {
            if(path.second==nodes[origin].Tdata[i][path.first])
            {
                for(int j=0;j<nodes[origin].Tdata[i].size();j++)
                {
                    tdata[k].push_back(nodes[origin].Tdata[i][j]);
                }
                k++;
            }
        }

    }

    tdata.resize(k);

    return tdata;
}

void setTdata_for_new_branches(const int& colnum,const int& origin,bool shuffleagain,time_t seed)
{
    unordered_set<string> subfeature = Get_featurearr(colnum,origin);
    vector<vector<double>> tdatareg;
    for(auto& x : subfeature)
    {
        pair<int,string> path;

        path = make_pair(colnum,x);

        vector<vector<string>> tdata = getTdata(path,origin);
        vector<int> coldrop = {colnum};
        column_drop(coldrop,tdata);
        if(shuffleagain)
        {
            shuffle_rows(tdata,seed);
        }

        add_node("",tdata,path,0,0,0,tdatareg);

        add_rules(origin,nodes.size()-1,x);

    }

}

void MakeDecisions(vector<vector<string>>& tdata,set<string>& targets,const double& single_node_accuracy_per,bool shuffleagain,time_t seed)
{
    if(nodes.empty())
    {
        vector<vector<double>> tdatareg;
        pair<int,string> path;
        add_node("",tdata,path,0,0,0,tdatareg);
        int colnum;
        string attr = set_node(colnum,0,targets,100);
        nodes[0].Attr = attr;
        nodes[0].colnum = colnum;
        setTdata_for_new_branches(colnum,0,shuffleagain,seed);
        nodes[0].Tdata.clear();

        MakeDecisions(tdata,targets,single_node_accuracy_per,shuffleagain,seed);

    }
    else
    {
        for(int i=0;i<nodes.size();i++)
        {
            if(nodes[i].Attr=="")
            {
                int colnum;
                string attr = set_node(colnum,i,targets,single_node_accuracy_per);

                nodes[i].Attr = attr;
                nodes[i].colnum = colnum;

                if(colnum!=10000)
                {
                    setTdata_for_new_branches(colnum,i,shuffleagain,seed);
                }

                nodes[i].Tdata.clear();

            }

        }

    }

}

set<string> findTargets(vector<vector<string>>& tdata)
{
    set<string> targets;

    for(int i=1;i<tdata.size();i++)
    {
        if(tdata[i][tdata[i].size()-1]!=" " && tdata[i][tdata[i].size()-1]!="")
        {
            targets.insert(tdata[i][tdata[i].size()-1]);
        }

    }

    return targets;
}

double calGniimpurityforNumericaldata(vector<vector<string>>& tdata,const double& threshold,const int& j,set<string>& targets)
{
    vector<double> Hfirstcountarr;
    vector<double> Lfirstcountarr;

    for(auto& x : targets)
    {
        double Hfirstcount = 0;
        double Lfirstcount = 0;
        for(int i=1;i<tdata.size();i++)
        {
            if(atof(tdata[i][j].c_str())>=threshold)
            {
                if(tdata[i][tdata[0].size()-1]==x)
                {
                    Hfirstcount++;
                }

            }
            else
            {
                if(tdata[i][tdata[0].size()-1]==x)
                {
                    Lfirstcount++;
                }

            }

        }
        Hfirstcountarr.push_back(Hfirstcount);
        Lfirstcountarr.push_back(Lfirstcount);
    }

    double Hcounttotal =accumulate(Hfirstcountarr.begin(),Hfirstcountarr.end(),0.0);
    double Lcounttotal =accumulate(Lfirstcountarr.begin(),Lfirstcountarr.end(),0.0);
    double valH = 1;
    double valL = 1;
    for(int i=0;i<Hfirstcountarr.size();i++)
    {
        valH-=pow((Hfirstcountarr[i]/Hcounttotal),2);
    }
    for(int i=0;i<Lfirstcountarr.size();i++)
    {
        valL-=pow((Lfirstcountarr[i]/Lcounttotal),2);
    }

    double sumofGini = (valH*((Hcounttotal)/(tdata.size()-1)))+(valL*((Lcounttotal)/(tdata.size()-1)));

    return sumofGini;

}

pair<double,double> find_the_threshold(vector<vector<string>>& tdata,const int& j,vector<double>& vec,set<string>& targets)
{
    map<double,double> thresholds;

    for(int i=0;i<vec.size();i++)
    {
        double gini = calGniimpurityforNumericaldata(tdata,vec[i],j,targets);
        if(!isnan(gini))
        {
            thresholds.insert(make_pair(gini,vec[i]));
        }

    }

    return *next(thresholds.begin(),0);
}

vector<pair<int,double>> seperate_num_cols_find_threshold(vector<vector<string>>& tdata,set<string>& targets,vector<int>& colwithnums)
{
    typedef vector<pair<int,vector<double>>> Dtype;

    Dtype allpairs;
    vector<double> vec;
    for(auto& x : colwithnums)
    {
        for(int i=1;i<tdata.size();i++)
        {
            vec.push_back(atof(tdata[i][x].c_str()));
        }
        sort(vec.begin(),vec.end());
        allpairs.push_back(make_pair(x,vec));
        vec.clear();
    }
    vector<pair<int,double>> relation;
    for(auto& x : allpairs)
    {
        pair<double,double> threshold = find_the_threshold(tdata,x.first,x.second,targets);

        for(int i=1;i<tdata.size();i++)
        {
            if(atof(tdata[i][x.first].c_str())>=threshold.second)
            {
                tdata[i][x.first] = "1";
            }
            else
            {
                tdata[i][x.first] = "0";
            }
        }
        relation.push_back(make_pair(x.first,threshold.second));
    }
    return relation;

}
template<typename T>
void cal_Accuracy(vector<T>& prediction,vector<T> Actuals)
{
    double correct = 0;
    for(int i=1;i<prediction.size();i++)
    {
        if(prediction[i]==Actuals[i])
        {
            correct++;
        }
    }

    cout<<"Accuracy Score: "<<((correct/prediction.size())*100)<<" %"<<endl;
}

vector<string> Predict_outcome(vector<vector<string>>& input,set<string>& targets,vector<string>& header,bool showR,bool hasheader)
{
    srand((unsigned)time(0));
    //string first = *next(targets.begin(),0);
    //string second = *next(targets.begin(),1);
    vector<string> verdiction;
    for(int q=0;q<input.size();q++)
    {
        int colnum = nodes[0].colnum;
        string rule = input[q][colnum];

        int index = 0;

        while(1)
        {
            bool check = false;
            for(int j=0;j<sizeof(adj_rules)/sizeof(adj_rules[0]);j++)
            {
                if(rule==adj_rules[index][j])
                {
                    index = j;
                    check = true;
                    break;
                }

            }
            bool check2 = false;
            for(auto& x : targets)
            {
                if(nodes[index].Attr==x)
                {
                    if(showR)
                    {
                        cout<<"Prediction: "<<x<<endl;
                    }
                    check2 = true;
                    verdiction.push_back(x);
                    break;
                }
            }
            if(check2)
            {
                break;
            }

            if(!check)
            {
                if(hasheader)
                {
                    if(q!=0)
                    {
                        int in = (rand()%targets.size());
                        string pre = *next(targets.begin(),in);
                        if(showR)
                        {
                            cout<<"No Prediction,but chose randomly: "<<pre<<endl;
                        }

                        verdiction.push_back(pre);
                    }
                }
                else
                {
                    int in = (rand()%targets.size());
                    string pre = *next(targets.begin(),in);
                    if(showR)
                    {
                        cout<<"No Prediction,but chose randomly: "<<pre<<endl;
                    }

                    verdiction.push_back(pre);
                }

                break;
            }
            for(int i=0;i<header.size();i++)
            {
                if(header[i]==nodes[index].Attr)
                {
                    rule = input[q][i];
                    break;
                }
            }

        }
    }

    return verdiction;
}


vector<pair<int,double>> build_Decision_tree(vector<vector<string>>& tdata,vector<int>& colwithnums,const double& split_train_per,const double& single_node_accuracy_per,bool shuffleagain,time_t seed)
{
    set<string> targets = findTargets(tdata);
    vector<pair<int,double>> rel = seperate_num_cols_find_threshold(tdata,targets,colwithnums);

    vector<vector<string>> testdatasplited;
    vector<string> actuals;
    const int testdataN = (tdata.size())-((split_train_per*(tdata.size()))/100);
    testdatasplited.resize(testdataN);
    int y=0;

    vector<string> header;
    for(int j=0;j<tdata[0].size()-1;j++)
    {
        header.push_back(tdata[0][j]);
    }

    for(int i=tdata.size()-testdataN;i<tdata.size();i++)
    {
        actuals.push_back(tdata[i][tdata[i].size()-1]);
        for(int j=0;j<tdata[i].size();j++)
        {
            testdatasplited[y].push_back(tdata[i][j]);
        }
        y++;

    }

    tdata.resize(tdata.size()-testdataN);
    tdata.shrink_to_fit();

    vector<int> drops = {(int(testdatasplited[0].size())-1)};
    column_drop(drops,testdatasplited);

    MakeDecisions(tdata,targets,single_node_accuracy_per,shuffleagain,seed);
    cout<<"Model is ready to predict!......"<<endl;
    vector<string> predicitons = Predict_outcome(testdatasplited,targets,header,true,false);
    cout<<"Model_";
    cal_Accuracy(predicitons,actuals);

    return rel;
}


bool isinclude(const string& str)
{
    for(int k=0;k<str.length();k++)
    {
        if(str[k]=='\"')
        {
            return true;
        }
    }
    return false;
}


void drop_Nan_row(vector<vector<string>>& tdata)
{
    for(int i=0;i<tdata.size();i++)
    {
        for(int j=0;j<tdata[i].size();j++)
        {
            if(tdata[i][j]=="Nan" || tdata[i][j]=="nan" || tdata[i][j]=="NAN")
            {
                tdata[i].clear();

            }
        }
    }
}

pair<int,string> get_each_freq(const int& k,vector<vector<string>>& tdata)
{
    map<string,int> mp;

    for(int t=0;t<tdata.size();t++)
    {
        mp[tdata[t][k]]++;

    }
    map<int,string> vecp;

    for(auto& x : mp)
    {
        vecp.insert(make_pair(x.second,x.first));
    }

    return *next(vecp.begin(),vecp.size()-1);
}

void replace_columnwithfreq(vector<int>& freqedtion,vector<vector<string>>& tdata)
{
    for(int k=0;k<freqedtion.size();k++)
    {
        pair<int,string> mostfreq = get_each_freq(freqedtion[k],tdata);

        for(int t=0;t<tdata.size();t++)
        {
            if(tdata[t][freqedtion[k]]=="B28" || tdata[t][freqedtion[k]]==" ")
            {
                tdata[t][freqedtion[k]] = mostfreq.second;
            }
        }
    }
}

void featurize_data(vector<vector<string>>& tdata,vector<int>& alphafeature)
{
    for(int j=0;j<alphafeature.size();j++)
    {
        vector<string> data;
        for(int i=1;i<tdata.size();i++)
        {
            data.push_back(tdata[i][alphafeature[j]]);
        }
        sort(data.begin(),data.end());
        data.erase(unique(data.begin(),data.end()),data.end());

        for(int c=0;c<data.size();c++)
        {
            for(int i=1;i<tdata.size();i++)
            {
                if(tdata[i][alphafeature[j]]==data[c])
                {
                    tdata[i][alphafeature[j]] = InttoString(c);
                }
            }
        }

    }

}

void sumX(const double& x,double& SXv)
{
    SXv += x;
}

void sumY(const double& y,double& SYv)
{
    SYv += y;
}

void sumXY(const double& x,const double& y,double& SXYv)
{
    SXYv += x*y;
}

void sumX2(const double& x,double& SX2v)
{
    SX2v += pow(x,2);
}

void sumY2(const double& y,double& SY2v)
{
    SY2v += pow(y,2);
}

double get_cor_coe(int t,const int& x,vector<vector<string>>& tdata,const int& y,vector<int>& alphafeature)
{
    if(t==0)
    {
        featurize_data(tdata,alphafeature);
    }

    double sumXv=0,sumYv=0,sumX2v=0,sumY2v=0,sumXYv=0;
    for(int i=1;i<tdata.size();i++)
    {
        sumX(atof(tdata[i][x].c_str()),sumXv);
        sumY(atof(tdata[i][y].c_str()),sumYv);
        sumX2(atof(tdata[i][x].c_str()),sumX2v);
        sumY2(atof(tdata[i][y].c_str()),sumY2v);
        sumXY(atof(tdata[i][x].c_str()),atof(tdata[i][y].c_str()),sumXYv);

    }
    double upper = (tdata.size()*sumXYv)-(sumXv*sumYv);
    double lower = sqrt(((tdata.size()*sumX2v)-pow(sumXv,2))*((tdata.size()*sumY2v)-pow(sumYv,2)));
    double corcoe = upper/lower;

    return corcoe;
}

vector<pair<int,int>> get_best_corrcoecolms(vector<int>& tocorcoe,vector<vector<string>> tdata,vector<int>& alphafeature)
{
    vector<pair<int,int>> mp;
    double minval = 10000;
    int mincolx = 0;
    int mincoly = 0;
    int t=0;
    for(auto& x : tocorcoe)
    {
        for(int y=0;y<tdata[0].size();y++)
        {
            double corcoe = get_cor_coe(t++,x,tdata,y,alphafeature);

            if(corcoe>=0)
            {
                double abslD = 1-corcoe;
                if(minval>abslD && abslD!=0)
                {
                    minval = abslD;
                    mincolx = x;
                    mincoly = y;
                }
            }
            else
            {
                double abslD = 1-abs(corcoe);
                if(minval>abslD & abslD!=0)
                {
                    minval = abslD;
                    mincolx = x;
                    mincoly = y;
                }
            }

        }
        mp.push_back(make_pair(mincolx,mincoly));

    }

    return mp;

}

vector<vector<string>> fixmissingnumvalue(vector<int>& tocorcoe,vector<vector<string>> tdata,vector<int>& alphafeature)
{
   vector<pair<int,int>> mp = get_best_corrcoecolms(tocorcoe,tdata,alphafeature);
   vector<pair<int,vector<string>>> data;

   for(auto& x : mp)
   {
       vector<string> fixed;
       vector<string> mis;
       for(int i=1;i<tdata.size();i++)
       {
           if(tdata[i][x.first]==" ")
           {
               if(tdata[i][x.second]!=" ")
               {
                   mis.push_back(tdata[i][x.second]);
               }

           }
       }
       for(int r=0;r<mis.size();r++)
       {
           double sum=0;
           int v =0;
           for(int i=1;i<tdata.size();i++)
           {
               if(tdata[i][x.second]==mis[r])
               {
                   if(tdata[i][x.first]!=" ")
                   {
                       sum += atof(tdata[i][x.first].c_str());
                       v++;
                   }
               }
           }

           fixed.push_back(InttoString(sum/v));

       }

       map<string,int> freq;
       for(int u=0;u<fixed.size();u++)
       {
            freq[fixed[u]]++;
       }
       map<int,string> collected;
       for(auto& z : freq)
       {
           collected.insert(make_pair(z.second,z.first));
       }

        for(int u=0;u<fixed.size();u++)
       {
           if(fixed[u]=="nan" || fixed[u]=="0")
           {
               fixed[u] = (*next(collected.begin(),collected.size()-1)).second;
           }
       }
       data.push_back(make_pair(x.first,fixed));
   }


   for(auto& x : data)
   {
       int h=1;
       for(int j=0;j<x.second.size();j++)
       {
            for(int i=h;i<tdata.size();i++)
            {
                if(tdata[i][x.first]==" ")
                {
                    tdata[i][x.first] = x.second[j];
                    h = i;
                    break;
                }
            }
       }

   }

    return tdata;

}
template<typename T>
void reposition_target_col(vector<vector<T>>& tdata,const int& current)
{
    for(int i=0;i<tdata.size();i++)
    {
        string td = tdata[i][current];
        tdata[i].erase(tdata[i].begin()+current);
        tdata[i].push_back(td);
    }

}

vector<vector<string>> readprepareTraindataset(const char* fname)
{
    vector<string> data;
    vector<vector<string>> tdata;
    ifstream file(fname);
    string line = "";

    while(getline(file,line))
    {
       for(int i=0;i+1<line.length();i++)
       {
           if(line[i]==',' && line[i+1]==',')
           {
               line.insert(i+1," ");
           }
       }
        data.push_back(line);
    }

    file.close();

    tdata.resize(data.size());

    for(int i=0;i<data.size();i++)
    {
        vector<string> str = split(data[i],",");
        for(int j=1;j<str.size();j++)
        {
            if(!isinclude(str[j]) && str[j]!="Name" && str[j]!="PassengerId")
            {
                tdata[i].push_back(str[j]);
            }

        }
    }

    vector<int> drops = {1,6,7};
    column_drop(drops,tdata);
    drop_Nan_row(tdata);
    vector<int> freqedtion = {6};
    replace_columnwithfreq(freqedtion,tdata);
    reposition_target_col(tdata,0);
    vector<int> tocorcoe = {1};
    vector<int> alphafeature = {0,5};
    tdata = fixmissingnumvalue(tocorcoe,tdata,alphafeature);
    int q =0;
    for(int i=0;i<tdata.size();i++)
    {
        if(tdata[i].size()!=tdata[0].size())
        {
            drop_rows(tdata,i);
            q++;
        }
    }
    vector<vector<string>> tdatanew;
    tdatanew.resize(tdata.size()-q);

    int h=0;
    for(int i=0;i<tdata.size();i++)
    {
        if(tdata[i].size()!=0)
        {
            for(int j=0;j<tdata[i].size();j++)
            {
                tdatanew[h].push_back(tdata[i][j]);
            }
            h++;
        }

    }

    tdata.clear();

    return tdatanew;

}

vector<vector<string>> readprepareTestdataset(const char* fname,vector<pair<int,double>>& relation)
{
    vector<string> data;
    vector<vector<string>> tdata;
    ifstream file(fname);
    string line = "";

    while(getline(file,line))
    {
       for(int i=0;i+1<line.length();i++)
       {
           if(line[i]==',' && line[i+1]==',')
           {
               line.insert(i+1," ");
           }
       }
        data.push_back(line);
    }

    file.close();

    tdata.resize(data.size());

    for(int i=0;i<data.size();i++)
    {
        vector<string> str = split(data[i],",");
        for(int j=1;j<str.size();j++)
        {
            if(!isinclude(str[j]) && str[j]!="Name" && str[j]!="PassengerId")
            {
                tdata[i].push_back(str[j]);
            }

        }
    }

    vector<int> drops = {0,5,6};
    column_drop(drops,tdata);
    drop_Nan_row(tdata);
    vector<int> freqedtion = {5};
    replace_columnwithfreq(freqedtion,tdata);
    vector<int> tocorcoe = {1};
    vector<int> alphafeature = {0,5};
    tdata = fixmissingnumvalue(tocorcoe,tdata,alphafeature);
    int q =0;
    for(int i=0;i<tdata.size();i++)
    {
        if(tdata[i].size()!=tdata[0].size())
        {
            drop_rows(tdata,i);
            q++;
        }
    }
    vector<vector<string>> tdatanew;
    tdatanew.resize(tdata.size()-q);

    int h=0;
    for(int i=0;i<tdata.size();i++)
    {
        if(tdata[i].size()!=0)
        {
            for(int j=0;j<tdata[i].size();j++)
            {
                tdatanew[h].push_back(tdata[i][j]);
            }
            h++;
        }

    }

    tdata.clear();

    for(auto& x : relation)
    {
        for(int i=1;i<tdatanew.size();i++)
        {
            if(atof(tdatanew[i][x.first].c_str())>=x.second)
            {
                tdatanew[i][x.first] = "1";
            }
            else
            {
                tdatanew[i][x.first] = "0";
            }
        }

    }

    return tdatanew;
}

vector<double> CalculateThresholdReg(const int& index,const int& j)
{
    vector<double> thresholds;
    vector<double> colvals;

    for(int i=0;i<nodes[index].Tdatareg.size();i++)
    {
        colvals.push_back(nodes[index].Tdatareg[i][j]);
    }
    sort(colvals.begin(),colvals.end());

    double reductoin =0;
    for(int i=0;i+1<colvals.size();i++)
    {
        reductoin += colvals[i+1]-colvals[i];
    }

    reductoin = reductoin/(colvals.size()-1);

    double value=colvals[0];
    for(int i=0;i<colvals.size();i++)
    {
        thresholds.push_back(value);
        if(colvals[colvals.size()-1]<value)
        {
            break;
        }

        value+=reductoin;

    }

    return thresholds;
}

double cal_Sum_mse(const int& index,const int& j,const double& threshold)
{
    vector<double> lowvals;
    vector<double> highvals;
    for(int i=0;i<nodes[index].Tdatareg.size();i++)
    {
        if(threshold>=nodes[index].Tdatareg[i][j])
        {
            lowvals.push_back(nodes[index].Tdatareg[i][nodes[index].Tdatareg[i].size()-1]);
        }
        else
        {
            highvals.push_back(nodes[index].Tdatareg[i][nodes[index].Tdatareg[i].size()-1]);
        }
    }
    double Lsum=0,Hsum=0,LsumAve=0,HsumAve=0,Lsmse=0,Hsmse=0;

    Lsum = accumulate(lowvals.begin(),lowvals.end(),0.0);
    LsumAve = Lsum/lowvals.size();
    Hsum = accumulate(highvals.begin(),highvals.end(),0.0);
    HsumAve = Hsum/highvals.size();

    for(int i=0;i<lowvals.size();i++)
    {
        Lsmse+=pow((lowvals[i]-LsumAve),2);
    }

    for(int i=0;i<highvals.size();i++)
    {
        Hsmse+=pow((highvals[i]-HsumAve),2);
    }

    return Lsmse+Hsmse;

}

double proper_Mse(const int& index,const int& j,vector<double>& Lthresholds,double& bestthreshold)
{
    map<double,double> hold;

    for(int i=0;i<Lthresholds.size();i++)
    {
        double Smse = cal_Sum_mse(index,j,Lthresholds[i]);

        hold.insert(make_pair(Smse,Lthresholds[i]));
    }

    bestthreshold = (*next(hold.begin(),0)).second;

    return (*next(hold.begin(),0)).first;
}

void setTdata_for_new_branches_reg(const int& colnum,const int& origin,const double& threshold)
{
    vector<vector<string>> tdata;
    pair<int,string> path;
    vector<vector<double>> Truetdatareg;
    vector<vector<double>> Falsetdatareg;

    Truetdatareg.resize(nodes[origin].Tdatareg.size());
    Falsetdatareg.resize(nodes[origin].Tdatareg.size());

    int k=0;
    int n=0;
    bool check = false;
    for(int i=0;i<nodes[origin].Tdatareg.size();i++)
    {
        for(int j=0;j<nodes[origin].Tdatareg[i].size();j++)
        {
            if(nodes[origin].Tdatareg[i][colnum]<=threshold)
            {
                Truetdatareg[k].push_back(nodes[origin].Tdatareg[i][j]);
                check = true;
            }
            else
            {
                Falsetdatareg[n].push_back(nodes[origin].Tdatareg[i][j]);
                check = false;
            }
        }
        if(check)
        {
            k++;
        }
        else
        {
            n++;
        }
    }

    Truetdatareg.resize(k);
    Falsetdatareg.resize(n);

    add_node("",tdata,path,0,0,0,Truetdatareg);
    add_rules(origin,nodes.size()-1,"True");

    add_node("",tdata,path,0,0,0,Falsetdatareg);
    add_rules(origin,nodes.size()-1,"False");


}

string set_node_reg(int& colnum,const int& index,double& threshold,double& mse,double& average,vector<string>& header,const int& NumberofCases)
{
    map<double,pair<int,double>> hold;

    for(int j=0;j<nodes[index].Tdatareg[0].size()-1;j++)
    {
        vector<double> Lthresholds = CalculateThresholdReg(index,j);

        double bestthreshold;
        double Lmse = proper_Mse(index,j,Lthresholds,bestthreshold);
        pair<int,double> co = make_pair(j,bestthreshold);
        hold.insert(make_pair(Lmse,co));

    }
    mse = (*next(hold.begin(),0)).first;
    threshold = ((*next(hold.begin(),0)).second).second;

    vector<double> vals;
    for(int i=0;i<nodes[index].Tdatareg.size();i++)
    {
        vals.push_back(nodes[index].Tdatareg[i][nodes[index].Tdatareg[i].size()-1]);
    }

    average = accumulate(vals.begin(),vals.end(),0.0)/vals.size();
    colnum = ((*next(hold.begin(),0)).second).first;
    if(vals.size()<=NumberofCases)
    {
        colnum = 10000;
        return "result";
    }
    else
    {
        return header[((*next(hold.begin(),0)).second).first];
    }

}

void MakeregressionDecision(vector<vector<double>>& tdatareg,vector<string>& header,const int& NumberofCases)
{
    if(nodes.empty())
    {
        vector<vector<string>> tdata;
        pair<int,string> path;

        add_node("",tdata,path,0,0,0,tdatareg);

        int colnum;
        double threshold;
        double mse;
        double average;

        string attr = set_node_reg(colnum,0,threshold,mse,average,header,0);

        nodes[0].Attr = attr;
        nodes[0].colnum = colnum;
        nodes[0].Threshold = threshold;
        nodes[0].Average = average;
        nodes[0].MSE = mse;

        setTdata_for_new_branches_reg(colnum,0,threshold);

        nodes[0].Tdatareg.clear();

        MakeregressionDecision(tdatareg,header,NumberofCases);

    }
    else
    {
        for(int i=0;i<nodes.size();i++)
        {
            if(nodes[i].Attr=="")
            {
                int colnum;
                double threshold;
                double mse;
                double average;

                string attr = set_node_reg(colnum,i,threshold,mse,average,header,NumberofCases);

                nodes[i].Attr = attr;
                nodes[i].colnum = colnum;
                nodes[i].Threshold = threshold;
                nodes[i].Average = average;
                nodes[i].MSE = mse;

                if(colnum!=10000)
                {
                    setTdata_for_new_branches_reg(colnum,i,threshold);

                }

                nodes[i].Tdatareg.clear();

            }

        }

    }
}

vector<double> Predict_outcome_reg(vector<vector<double>>& input,vector<string>& header,bool showR)
{
    vector<double> verdiction;
    for(int q=0;q<input.size();q++)
    {
        int index = 0;
        int colnum = nodes[0].colnum;
        int threshold = nodes[0].Threshold;
        double val = input[q][colnum];
        string rule = "";

        if(val<=threshold)
        {
            rule = "True";
        }
        else
        {
            rule = "False";
        }

        while(1)
        {
            bool check = false;
            for(int j=0;j<sizeof(adj_rules)/sizeof(adj_rules[0]);j++)
            {
                if(rule==adj_rules[index][j])
                {
                    index = j;
                    check = true;
                    break;
                }

            }

            if(nodes[index].Attr=="result")
            {
                if(showR)
                {
                    cout<<"Prediction: "<<nodes[index].Average<<endl;
                }

                verdiction.push_back(nodes[index].Average);
                break;
            }

            if(!check)
            {
                if(showR)
                {
                    cout<<"No Prediction! "<<endl;
                }
                verdiction.push_back(0.0);
                break;
            }
            for(int i=0;i<header.size()-1;i++)
            {
                if(header[i]==nodes[index].Attr)
                {
                    val = input[q][i];
                    threshold = nodes[index].Threshold;
                    if(val<=threshold)
                    {
                        rule = "True";
                    }
                    else
                    {
                        rule = "False";
                    }
                    break;
                }
            }

        }
    }

    return verdiction;
}

void cal_Rsquared_Outcomes(vector<double>& prediction,vector<double>& actuals)
{
    double RSS=0,TSS=0,R2=0;
    double ybar = accumulate(actuals.begin(),actuals.end(),0.0)/actuals.size();
    for(int i=0;i<prediction.size();i++)
    {
        RSS+=pow((log(actuals[i])-log(prediction[i])),2);
        TSS+=pow((log(actuals[i])-log(ybar)),2);
    }

    R2 = 1-(RSS/TSS);
    cout<<"R2: "<<R2<<endl;
}

void cal_RMSE_Outcomes(vector<double>& prediction,vector<double>& actuals)
{
    double rmse = 0;

    for(int i=0;i<prediction.size();i++)
    {
        rmse+=pow((log(prediction[i])-log(actuals[i])),2);
    }

    cout<<"RMSE: "<<sqrt(rmse/prediction.size())<<endl;
}

void BuildRegressionTree(vector<vector<double>>& tdata,vector<string>& header,const int& NumberofCases,const double& split_train_per)
{
    vector<vector<double>> testdatasplited;
    vector<double> actuals;
    const int testdataN = (tdata.size())-((split_train_per*(tdata.size()))/100);
    testdatasplited.resize(testdataN);
    int y=0;

    for(int i=tdata.size()-testdataN;i<tdata.size();i++)
    {
        actuals.push_back(tdata[i][tdata[i].size()-1]);
        for(int j=0;j<tdata[i].size();j++)
        {
            testdatasplited[y].push_back(tdata[i][j]);
        }
        y++;

    }

    tdata.resize(tdata.size()-testdataN);
    tdata.shrink_to_fit();

    vector<int> drops = {(int(testdatasplited[0].size())-1)};
    column_drop(drops,testdatasplited);

    MakeregressionDecision(tdata,header,NumberofCases);
    cout<<"Model is ready to predict!......"<<endl;
    vector<double> predicitons = Predict_outcome_reg(testdatasplited,header,false);
    cout<<"Model_";
    cal_RMSE_Outcomes(predicitons,actuals);
    cout<<"Model_";
    cal_Rsquared_Outcomes(predicitons,actuals);

}


vector<vector<double>> readprepareTraindatasetReg(const char* fname,vector<string>& header)
{
    vector<string> data;
    vector<vector<double>> tdatareg;
    ifstream file(fname);
    string line = "";
    int u=0;
    while(getline(file,line))
    {
        if(line!="")
        {
            for(int i=0;i+1<line.length();i++)
            {
                if(line[i]==',' && line[i+1]==',')
                {
                    line.insert(i+1," ");
                }
            }
                data.push_back(line);
                u++;
        }
        if(u==6000)
        {
            break;
        }

    }

    file.close();

    tdatareg.resize(data.size()-1);

    for(int i=0;i<data.size();i++)
    {
        vector<string> str = split(data[i],",");

        for(int j=0;j<str.size();j++)
        {
            if(i==0)
            {
                header.push_back(str[j]);
            }
            else
            {
                tdatareg[i-1].push_back(atof(str[j].c_str()));

            }

        }
    }


    vector<int> drops = {0};
    column_drop(drops,tdatareg);
    header_col_Drop(drops,header);

    return tdatareg;
}

double cal_stdsub(const string& sub,const int& j,const int& index)
{
    double Tstd = 0;
    vector<double> vals;
    for(int i=0;i<nodes[index].Tdata.size();i++)
    {
        if(nodes[index].Tdata[i].size()!=0)
        {
            if(nodes[index].Tdata[i][j]==sub)
            {
                vals.push_back(atof(nodes[index].Tdata[i][nodes[index].Tdata[0].size()-1].c_str()));

            }
        }

    }

    double ave = accumulate(vals.begin(),vals.end(),0.0)/vals.size();

    for(int i=0;i<vals.size();i++)
    {
        Tstd+=pow(vals[i]-ave,2);
    }

    return sqrt(Tstd/vals.size());

}

double cal_Wstd(const int& j,const int& index)
{
    int tcount;
    unordered_map<string,double> Umap = count_Features(j,index,tcount);

    double sumstd = 0;
    for(auto& x : Umap)
    {
        sumstd += (cal_stdsub(x.first,j,index)*(x.second/tcount));
    }

    return sumstd;

}

double cal_IndexTstd(const int& index)
{
    vector<double> ave;
    double Tstd = 0;
    for(int i=0;i<nodes[index].Tdata.size();i++)
    {
        ave.push_back(atof(nodes[index].Tdata[i][nodes[index].Tdata[i].size()-1].c_str()));
    }
    double val = accumulate(ave.begin(),ave.end(),0.0)/ave.size();

    for(int i=0;i<ave.size();i++)
    {
        Tstd+=pow(ave[i]-val,2);
    }

    return sqrt(Tstd/ave.size());
}

string set_node_reg_std(int& colnum,const int& index,const double& numofcases,vector<string>& header)
{
    double high = 0;
    int feature_index = 0;
    for(int j=0;j<nodes[index].Tdata[0].size()-1;j++)
    {
        double stdR = cal_IndexTstd(index)-cal_Wstd(j,index);

        if(high<stdR)
        {
            high = stdR;
            feature_index = j;
        }
    }

    colnum = feature_index;
    if(numofcases>=nodes[index].Tdata.size())
    {
        vector<double> lcol;
        colnum = 10000;
        for(int i=0;i<nodes[index].Tdata.size();i++)
        {
            lcol.push_back(atof(nodes[index].Tdata[i][nodes[index].Tdata[i].size()-1].c_str()));
        }

        return InttoString((accumulate(lcol.begin(),lcol.end(),0.0)/lcol.size()));
    }
    else
    {
        return header[feature_index];
    }

}

void MakeregressionDecision_std(vector<vector<string>>& tdata,vector<string>& header,const int& NumberofCases,bool shuffleagain,time_t seed)
{
    if(nodes.empty())
    {
        vector<vector<double>> tdatareg;
        pair<int,string> path;

        add_node("",tdata,path,0,0,0,tdatareg);

        int colnum;
        string attr = set_node_reg_std(colnum,0,0,header);

        nodes[0].Attr = attr;
        nodes[0].colnum = colnum;

        setTdata_for_new_branches(colnum,0,shuffleagain,seed);

        nodes[0].Tdata.clear();

        MakeregressionDecision_std(tdata,header,NumberofCases,shuffleagain,seed);

    }
    else
    {
        for(int i=0;i<nodes.size();i++)
        {
            if(nodes[i].Attr=="")
            {
                int colnum;
                string attr = set_node_reg_std(colnum,i,NumberofCases,header);

                nodes[i].Attr = attr;
                nodes[i].colnum = colnum;

                if(colnum!=10000)
                {
                    setTdata_for_new_branches(colnum,i,shuffleagain,seed);

                }

                nodes[i].Tdata.clear();

            }

        }

    }
}

vector<double> Predict_outcome_reg_std(vector<vector<string>>& input,vector<string>& header,bool showR)
{
    vector<double> verdiction;
    for(int q=0;q<input.size();q++)
    {
        int colnum = nodes[0].colnum;
        string rule = input[q][colnum];

        int index = 0;

        while(1)
        {
            bool check = false;
            for(int j=0;j<sizeof(adj_rules)/sizeof(adj_rules[0]);j++)
            {
                if(rule==adj_rules[index][j])
                {
                    index = j;
                    check = true;
                    break;
                }

            }

            if(isdigit(nodes[index].Attr[0]) && isdigit(nodes[index].Attr[nodes[index].Attr.length()-1]))
            {
                if(showR)
                {
                    cout<<"Prediction: "<<nodes[index].Attr<<endl;
                }

                verdiction.push_back(atof(nodes[index].Attr.c_str()));
                break;
            }
            if(!check)
            {
                if(showR)
                {
                    cout<<"No Prediction!"<<endl;
                }
                verdiction.push_back(0);
                break;
            }
            for(int i=0;i<header.size();i++)
            {
                if(header[i]==nodes[index].Attr)
                {
                    rule = input[q][i];
                    break;
                }
            }

        }
    }

    return verdiction;
}

void BuildRegressionTree_using_std(vector<vector<string>>& tdata,vector<string>& header,const int& NumberofCases,const double& split_train_per,bool shuffleagain,time_t seed)
{
    vector<vector<string>> testdatasplited;
    vector<double> actuals;
    const int testdataN = (tdata.size())-((split_train_per*(tdata.size()))/100);
    testdatasplited.resize(testdataN);
    int y=0;

    for(int i=tdata.size()-testdataN;i<tdata.size();i++)
    {
        actuals.push_back(atof(tdata[i][tdata[i].size()-1].c_str()));
        for(int j=0;j<tdata[i].size();j++)
        {
            testdatasplited[y].push_back(tdata[i][j]);
        }
        y++;

    }

    tdata.resize(tdata.size()-testdataN);
    tdata.shrink_to_fit();

    vector<int> drops = {(int(testdatasplited[0].size())-1)};
    column_drop(drops,testdatasplited);

    MakeregressionDecision_std(tdata,header,NumberofCases,shuffleagain,seed);
    cout<<"Model is ready to predict!......"<<endl;
    vector<double> predicitons = Predict_outcome_reg_std(testdatasplited,header,true);
    cout<<"Model_";
    cal_RMSE_Outcomes(predicitons,actuals);
    cout<<"Model_";
    cal_Rsquared_Outcomes(predicitons,actuals);


}

vector<vector<string>> readprepareTraindatasetReg_std(const char* fname,vector<string>& header)
{
    vector<string> data;
    vector<vector<string>> tdatareg;
    ifstream file(fname);
    string line = "";
    int u=0;
    while(getline(file,line))
    {
        if(line!="")
        {
            for(int i=0;i+1<line.length();i++)
            {
                if(line[i]==',' && line[i+1]==',')
                {
                    line.insert(i+1," ");
                }
            }
                data.push_back(line);
                u++;
        }

    }

    file.close();

    tdatareg.resize(data.size()-1);

    for(int i=0;i<data.size();i++)
    {
        vector<string> str = split(data[i],",");

        for(int j=0;j<str.size();j++)
        {
            if(i==0)
            {
                header.push_back(str[j]);
            }
            else
            {
                remove_charss(str[j],"\"");
                tdatareg[i-1].push_back(str[j]);

            }

        }
    }
    int q=0;
    for(int i=0;i<tdatareg.size();i++)
    {
        if(tdatareg[i].size()!=tdatareg[0].size())
        {
            drop_rows(tdatareg,i);
            q++;
        }
    }
    vector<vector<string>> tdatanew;
    tdatanew.resize(tdatareg.size()-q);

    int h=0;
    for(int i=0;i<tdatareg.size();i++)
    {
        if(tdatareg[i].size()!=0)
        {
            for(int j=0;j<tdatareg[i].size();j++)
            {
                tdatanew[h].push_back(tdatareg[i][j]);
            }
            h++;
        }

    }
    tdatareg.clear();

    reposition_target_col(tdatanew,9);
    string s = header[9];
    header.erase(header.begin()+9);
    header.push_back(s);

    vector<int> drops = {0,2,4,6};
    column_drop(drops,tdatanew);
    header_col_Drop(drops,header);

    return tdatanew;
}

void encode_target_variable(vector<vector<string>>& tdata)
{
    set<string> mp;
    for(int i=1;i<tdata.size();i++)
    {
        mp.insert(tdata[i][tdata[i].size()-1]);
    }
    double y=1;
    for(auto& x : mp)
    {
        for(int i=1;i<tdata.size();i++)
        {
            if(tdata[i][tdata[i].size()-1]==x)
            {
                tdata[i][tdata[i].size()-1] = InttoString(y);
            }
        }
        y++;
    }

}

vector<vector<string>> readIrisdata(const char* fname)
{
    vector<string> data;
    vector<vector<string>> tdata;
    ifstream file(fname);
    string line = "";

    while(getline(file,line))
    {
        if(line!="")
        {
            for(int i=0;i+1<line.length();i++)
            {
                if(line[i]==',' && line[i+1]==',')
                {
                    line.insert(i+1," ");
                }
            }
            data.push_back(line);
        }

    }

    file.close();

    tdata.resize(data.size());

    for(int i=0;i<data.size();i++)
    {
        vector<string> str = split(data[i],",");
        for(int j=0;j<str.size();j++)
        {
            tdata[i].push_back(str[j]);
        }
    }

    encode_target_variable(tdata);
    shuffle_rows(tdata,time(0));

    return tdata;
}

int main()
{

    /*
    vector<string> header;
    vector<vector<double>> train_data = readprepareTraindatasetReg("housing.csv",header);
    BuildRegressionTree(train_data,header,5,80);
    vector<double> outcome = Predict_outcome_reg(input,header,true);
    */

    vector<vector<string>> train_data = readIrisdata("iris.data");
    vector<int> colwithnums = {0,1,2};
    //Note: lowering single_node_accuracy(to prune the tree/reduce overfitting) will give a better outcome for test dataset,but with a low model_accuracyscore
    vector<pair<int,double>> relation = build_Decision_tree(train_data,colwithnums,80,80,true,2);

/*
    vector<vector<string>> test_data = readprepareTestdataset("test.csv",relation);

    vector<string> header;
    for(int j=0;j<test_data[0].size();j++)
    {
        header.push_back(train_data[0][j]);
    }

    set<string> targets = findTargets(train_data);
    vector<string> predictions = Predict_outcome(test_data,targets,header,true,true);

    vector<string> actuals;
    ifstream file("gender_submission.csv");
    string key = "";
    int u=0;
    while(getline(file,key))
    {
        if(u>0)
        {
            vector<string> str = split(key,",");
            actuals.push_back(str[1]);
        }

        u++;
    }

    cout<<"For test data_";
    cal_Accuracy(predictions,actuals);
*/

    return 0;

}
