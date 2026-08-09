src=open('setup_b_extra_setups_optimizer.py','r',encoding='utf-8').read()
src=src.replace("(df.dt.to_numpy()>=START.to_datetime64())&(df.dt.to_numpy()<END.to_datetime64())", "(df.dt.ge(START).to_numpy())&(df.dt.lt(END).to_numpy())")
src=src.replace("dt=r.dt,day=r.trade_day", "dt=r['dt'],day=r['trade_day']")
exec(compile(src,'setup_b_extra_setups_optimizer.py','exec'))
