"""Actual per-attempt stage clocks; overlapping stages must not be summed as wall time."""
from contextlib import contextmanager
from datetime import datetime,timezone
from decimal import Decimal
import time
from uuid import uuid4

from psycopg.types.json import Jsonb

from .domain import clean,ident


class AttemptTiming:
    def __init__(self,core,token):self.core,self.token=core,ident(token)

    def record(self,stage,start,end,elapsed,*,window=-1,details=None):
        if not stage.isidentifier() or len(stage)>64 or window< -1:
            raise ValueError('Bounded timing stage required')
        value=Decimal(str(elapsed)).quantize(Decimal('.000001'))
        if not value.is_finite() or value<0:raise ValueError('Invalid elapsed time')
        with self.core.db.transaction() as conn:
            conn.execute('INSERT INTO hj_timeline_timing(customer_id,attempt_token,stage,window_index,started_at,finished_at,elapsed_seconds,details) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                (self.core.customer,self.token,stage,window,start,end,value,Jsonb(clean(details or {}))))

    @contextmanager
    def stage(self,name,*,window=-1,details=None):
        start=datetime.now(timezone.utc);tick=time.monotonic();state=dict(outcome='returned')
        try:yield state
        except Exception:
            state['outcome']='failed'
            raise
        finally:
            self.record(name,start,datetime.now(timezone.utc),time.monotonic()-tick,window=window,details=dict(details or {},**state))


class PublicationTiming:
    def __init__(self,core,day,revision):
        self.core,self.day,self.revision=core,day,revision
        self.start,self.tick=datetime.now(timezone.utc),time.monotonic()
        self.stages={}

    @contextmanager
    def phase(self,name):
        tick=time.monotonic()
        try:yield
        finally:self.stages[name]=time.monotonic()-tick

    def __enter__(self):return self

    def __exit__(self,kind,error,traceback):
        with self.core.db.transaction() as conn:
            conn.execute('INSERT INTO hj_timeline_publication_timing(customer_id,id,day,revision,started_at,finished_at,elapsed_seconds,outcome,stages,error_code) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                (self.core.customer,uuid4(),self.day,self.revision,self.start,datetime.now(timezone.utc),
                 Decimal(str(time.monotonic()-self.tick)).quantize(Decimal('.000001')),
                 'failed' if error else 'completed',Jsonb(self.stages),kind.__name__ if kind else None))
