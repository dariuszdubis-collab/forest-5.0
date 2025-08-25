#property strict
#property description "ForestBridge: file bridge MT4 <-> Forest 5.0"

/*
 Attach this Expert Advisor to a chart with **AutoTrading enabled**
 and "Allow live trading" checked. All files are exchanged via the
 terminal data directory under `MQL4/Files/forest_bridge`.
 Subdirectories created on start:
   ticks    - latest tick in tick.json
   commands - incoming cmd_<id>.json
   results  - responses res_<id>.json
   state    - account and position snapshots
*/

input int    Slippage   = 3;
input int    Magic      = 50500;
input string BridgeRoot = "forest_bridge";   // MQL4/Files/forest_bridge
input bool   LogDebug   = true;

void Log(const string msg){ if(LogDebug) Print("[ForestBridge] ", msg); }
string JsonEscape(string s){ StringReplace(s,"\\","\\\\"); StringReplace(s,"\"","\\\""); return s; }
string Trim(const string s){ string r=s; while(StringLen(r)>0&&StringGetChar(r,0)<=32) r=StringSubstr(r,1);
 while(StringLen(r)>0&&StringGetChar(r,StringLen(r)-1)<=32) r=StringSubstr(r,0,StringLen(r)-1); return r; }
string ExtractStr(const string s,const string key,const string endq="\""){ int p=StringFind(s,key); if(p<0) return("");
 int a=p+StringLen(key); int b=StringFind(s,endq,a); if(b<0) b=StringLen(s); return StringSubstr(s,a,b-a); }
double ExtractDouble(const string s,const string key){ int p=StringFind(s,key); if(p<0) return(0.0);
 int a=p+StringLen(key); int b=StringFind(s,",",a); if(b<0) b=StringFind(s,"}",a); if(b<0) b=StringLen(s);
 return StrToDouble(Trim(StringSubstr(s,a,b-a))); }
string ErrToStr(const int code){ return IntegerToString(code); }
bool WriteJson(const string rel, const string json){
 int h=FileOpen(rel, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_SHARE_READ|FILE_SHARE_WRITE);
 if(h==INVALID_HANDLE){ Print("WriteJson open failed: ",rel," err=",GetLastError()); return false; }
 FileWriteString(h,json); FileClose(h); return true; }
string Join(const string a,const string b){ return a+"\\"+b; }

void EnsureDirs(){
 string sub[] = {"","ticks","commands","results","state"};
 for(int i=0;i<ArraySize(sub);i++){
   string p = BridgeRoot + (sub[i]=="" ? "" : ("\\"+sub[i]));
   if(!FolderCreate(p) && GetLastError()!=ERR_FILE_ALREADY_EXISTS)
      Print("Failed to create folder ", p, " err=", GetLastError());
 }
}

void WriteTick(){
 string p=Join(Join(BridgeRoot,"ticks"),"tick.json");
 string js=StringFormat("{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"time\":%d}", _Symbol,Bid,Ask,TimeCurrent());
 WriteJson(p,js);
}
void WriteState(){
 double eq=AccountEquity();
 WriteJson(Join(Join(BridgeRoot,"state"),"account.json"), StringFormat("{\"equity\":%.2f}",eq));
 double qty=0.0;
 for(int i=OrdersTotal()-1;i>=0;i--){
   if(!OrderSelect(i,SELECT_BY_POS,MODE_TRADES)) continue;
   if(OrderSymbol()!=_Symbol) continue;
   int t=OrderType();
   if(t==OP_BUY)  qty+=OrderLots();
   if(t==OP_SELL) qty-=OrderLots();
 }
string fn=StringFormat("position_%s.json",_Symbol);
WriteJson(Join(Join(BridgeRoot,"state"),fn), StringFormat("{\"qty\":%.2f}",qty));
}

void NormalizeStops(const string symbol,bool is_buy,double price,double &sl,double &tp){
 int digits=(int)MarketInfo(symbol,MODE_DIGITS);
 double point=MarketInfo(symbol,MODE_POINT);
 double mindist=MarketInfo(symbol,MODE_STOPLEVEL)*point;
 if(is_buy){
   if(sl>0){
     if(price-sl<mindist) sl=price-mindist;
     sl=NormalizeDouble(sl,digits);
   }
   if(tp>0){
     if(tp-price<mindist) tp=price+mindist;
     tp=NormalizeDouble(tp,digits);
   }
 } else {
   if(sl>0){
     if(sl-price<mindist) sl=price+mindist;
     sl=NormalizeDouble(sl,digits);
   }
   if(tp>0){
     if(price-tp<mindist) tp=price-mindist;
     tp=NormalizeDouble(tp,digits);
   }
 }
}

string ExecuteCommand(const string id,const string action,const string symbol,double volume,double sl,double tp){
 RefreshRates(); int ticket=-1; double price=0.0; bool ok=false; string err="";
 if(action=="BUY"){
   double lots=volume; double ask=NormalizeDouble(Ask,Digits);
   if(!MathIsValidNumber(sl) || !MathIsValidNumber(tp) || sl<=0 || tp<=0){
     err="invalid_stops";
   } else {
     double point=MarketInfo(symbol,MODE_POINT);
     double mindist=MarketInfo(symbol,MODE_STOPLEVEL)*point;
     if(ask-sl<mindist || tp-ask<mindist){
       err="invalid_stops";
     } else {
       double _sl=sl; double _tp=tp;
       NormalizeStops(symbol,true,ask,_sl,_tp);
       bool adjusted = (_sl!=sl) || (_tp!=tp);
       if(LogDebug && (_sl>0 || _tp>0)){
         if(adjusted)
           Log("BUY stops adjusted sl="+DoubleToString(sl,Digits)+"->"+DoubleToString(_sl,Digits)+
               " tp="+DoubleToString(tp,Digits)+"->"+DoubleToString(_tp,Digits));
         else
           Log("BUY stops sl="+DoubleToString(_sl,Digits)+" tp="+DoubleToString(_tp,Digits));
       }
       ticket=OrderSend(symbol,OP_BUY,lots,ask,Slippage,_sl,_tp,"FOREST",Magic,0,clrGreen);
       if(ticket>0){ ok=true; price=ask; } else { err=ErrToStr(GetLastError()); }
     }
   }
 } else if(action=="SELL"){
   double toclose=volume; ok=true;
   for(int i=OrdersTotal()-1;i>=0 && toclose>0;i--){
     if(!OrderSelect(i,SELECT_BY_POS,MODE_TRADES)) continue;
     if(OrderSymbol()!=symbol) continue;
     if(OrderType()!=OP_BUY) continue;
     double lot=OrderLots(); double part=MathMin(lot,toclose);
     double bid=NormalizeDouble(Bid,Digits);
     bool res=OrderClose(OrderTicket(),part,bid,Slippage,clrRed);
     if(!res){ ok=false; err=ErrToStr(GetLastError()); break; }
     toclose-=part; RefreshRates();
   }
   if(toclose>0 && ok){ ok=false; err="no position to sell"; }
   price=Bid;
 } else { err="unknown action"; }
 string st= ok ? "filled":"rejected"; string ejs= ok ? "null":("\""+JsonEscape(err)+"\"");
 return StringFormat("{\"id\":\"%s\",\"status\":\"%s\",\"ticket\":%d,\"price\":%.5f,\"error\":%s}",
                     JsonEscape(id), st, ticket, price, ejs);
}

void ProcessCommands(){
 string mask=Join(Join(BridgeRoot,"commands"),"cmd_*.json"); string fname;
 int hfind=(int)FileFindFirst(mask,fname); if(hfind==INVALID_HANDLE) return;
 do{
   string path=Join(Join(BridgeRoot,"commands"),fname);
   int h=FileOpen(path, FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_READ|FILE_SHARE_WRITE);
   if(h==INVALID_HANDLE){ Print("Cannot open cmd: ",path," err=",GetLastError()); continue; }
   string s=FileReadString(h); FileClose(h);
   string id=fname; int dot=StringFind(id,".json",0); if(dot>0) id=StringSubstr(id,0,dot);
   if(StringFind(id,"cmd_")==0) id=StringSubstr(id,4);
  string action=ExtractStr(s,"\"action\":\"","\"");
  string symbol=ExtractStr(s,"\"symbol\":\"","\""); if(symbol=="") symbol=_Symbol;
  double volume=ExtractDouble(s,"\"volume\":");
  double sl=ExtractDouble(s,"\"sl\":");
  double tp=ExtractDouble(s,"\"tp\":");
  if(LogDebug) Log("CMD "+id+" "+action+" "+symbol+" vol="+DoubleToString(volume,2)+
                     " sl="+DoubleToString(sl,Digits)+" tp="+DoubleToString(tp,Digits));
  string resjs=ExecuteCommand(id,action,symbol,volume,sl,tp);
  string resfile=Join(Join(BridgeRoot,"results"),"res_"+id+".json");
  WriteJson(resfile,resjs);
  FileDelete(path);
 } while(FileFindNext(hfind,fname));
 FileFindClose(hfind);
}

int OnInit(){ EnsureDirs(); EventSetTimer(1); Log("Ready: MQL4\\Files\\"+BridgeRoot); return(INIT_SUCCEEDED); }
void OnDeinit(const int reason){ EventKillTimer(); }
void OnTick(){  WriteTick(); }
void OnTimer(){ WriteState(); ProcessCommands(); }
