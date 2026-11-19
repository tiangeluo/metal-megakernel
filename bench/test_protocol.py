"""Protocol regression tests. All fake measurements live only inside TemporaryDirectory."""
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench.protocol import ATTENTION, FIXTURES, PILOT, SCHEMA, SCHEDULES, compare_attention
from bench.report import attach_trace_counts, intervals, load_sessions, read_case, write_report, metrics
from bench.run import preflight, randomized_order, run_case, save_case


class ProtocolTests(unittest.TestCase):
    def test_rejects_absent_gpu_before_importing_mlx(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"env.json"
            p.write_text(json.dumps({"gpu_probe":{"passed":False}}))
            with self.assertRaisesRegex(RuntimeError,"no successful native"):
                preflight(p)

    def test_checks_physical_cache_even_when_gated_output_matches(self):
        shape=(1,1,3,2)
        original={k:np.zeros(shape,np.float16) for k in ("cache_k","hist_k")}
        expected={k:v.copy() for k,v in original.items()}
        out=np.ones((1,1,2),np.float16)
        expected["cache_k"][:,:,1]=1
        measured={k:v.copy() for k,v in expected.items()}
        compare_attention((out,expected),(out,measured),original,1)
        measured["cache_k"][:,:,2,0]=1e-5
        with self.assertRaises(AssertionError):
            compare_attention((out,expected),(out,measured),original,1)

    def test_all_schedules_pair_and_fixtures_interleave(self):
        class Dummy:
            def reset(self):
                self.ready=True
            def step(self):
                if not self.ready:
                    raise ValueError("reset must precede every timing")
                self.ready=False
                return None,None
            def read(self):
                return "UNIT-TEST"
        closed=[]
        def prepare(case,seed):
            def check(result):
                self.assertEqual(result,"UNIT-TEST")
                return {"unit":0.0}
            return {m:Dummy() for m in case.methods},check,lambda:closed.append(seed)
        with patch("bench.run.adapters.prepare",side_effect=prepare):
            rows,parity=run_case(PILOT[0],"unit-only",1,8,1)
        self.assertEqual(len(rows),8*len(PILOT[0].methods))
        by_block={}
        for r in rows:
            by_block.setdefault(r["block"],[]).append(r)
        for data in by_block.values():
            self.assertEqual({r["method"] for r in data},set(PILOT[0].methods))
            self.assertEqual(len({r["fixture"] for r in data}),1)
        order=list(by_block)
        for i in range(0,len(order),2):
            self.assertEqual({by_block[order[i]][0]["fixture"],by_block[order[i+1]][0]["fixture"]},{0,1})
        self.assertEqual(set(parity),{0,1})
        self.assertEqual(set(closed),{1,2})

    def test_paired_bootstrap_preserves_per_block_scaling(self):
        x=np.linspace(20,120,100)
        arrays=[np.stack([x,x/2,x*2,x],axis=1),np.stack([x,x/2,x*2,x],axis=1)*1.2]
        _,_,speed,ci=intervals(arrays,np.random.default_rng(2),replicates=100)
        np.testing.assert_allclose(speed,[1,2,.5,1])
        np.testing.assert_allclose(ci,np.tile([1,2,.5,1],(2,1)))


def test_data(root, name):
    """Build a non-GPU artifact solely for structural tests in an ephemeral directory."""
    d=root/name/"attention"
    d.mkdir(parents=True)
    machine={"device":"UNIT-TEST FIXTURE ONLY","sources_sha256":"UNIT-TEST"}
    spec=dict(schema=SCHEMA,suite="attention",session=name,seed=42,trials=100,warmups=1,fixtures=FIXTURES,machine=machine)
    (d/"manifest.json").write_text(json.dumps({"spec":spec,"status":"complete","environment":{"gpu_probe":{"passed":True,"device":"UNIT-TEST FIXTURE ONLY"}}}))
    for case in ATTENTION:
        rng=random.Random(9)
        rows=[]
        for block in range(100):
            for order,m in enumerate(randomized_order(case.methods,rng)):
                rows.append({"session":name,"case":case.name,"fixture":block%2,"block":block,
                             "order":order,"method":m,"wall_us":5+case.methods.index(m)+block/1000,
                             "native_gpu_us":"","encoded_dispatches":""})
        proof={f:{m:{"output":0.0} for m in case.methods} for f in range(2)}
        save_case(d,case,rows,proof)
    return d,spec


# Avoid accidental pytest collection as a standalone test.
test_data.__test__=False


class ReportTests(unittest.TestCase):
    def test_two_distinct_sessions_and_digest_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory,spec=test_data(root,"unit-a")
            with self.assertRaisesRegex(ValueError,"two independent"):
                load_sessions(root,None,ATTENTION)
            test_data(root,"unit-b")
            sessions,_=load_sessions(root,None,ATTENTION)
            self.assertEqual(set(sessions),{"unit-a","unit-b"})
            csv_file=directory/(PILOT[0].name+".csv")
            csv_file.write_text(csv_file.read_text().replace("5.001","111.0",1))
            with self.assertRaisesRegex(ValueError,"proof"):
                read_case(directory,PILOT[0],spec)

    def test_trace_counts_require_a_named_existing_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            index=root/"trace-index.csv"
            index.write_text("case,method,observed_dispatches,trace_path\nattention-b1-l64,compiled,9,capture.trace\n")
            items=[{"case":"attention-b1-l64","method":"compiled"}]
            with self.assertRaisesRegex(ValueError,"missing local"):
                attach_trace_counts(items,index)
            (root/"capture.trace").mkdir()
            attach_trace_counts(items,index)
            self.assertEqual(items[0]["observed_dispatches"],9)

    def test_incomplete_or_unverified_sessions_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            d,_=test_data(root,"unit-a")
            test_data(root,"unit-b")
            path=d/"manifest.json"
            manifest=json.loads(path.read_text())
            manifest["status"]="incomplete"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"two independent"):
                load_sessions(root,None,ATTENTION)
            manifest["status"]="complete"
            manifest["environment"]["gpu_probe"]["passed"]=False
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"two independent"):
                load_sessions(root,None,ATTENTION)

    def test_table_and_plot_generation_using_discarded_mock_timings(self):
        try:
            import matplotlib  # noqa
        except ImportError:
            self.skipTest("optional Matplotlib unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            test_data(root,"unit-a")
            test_data(root,"unit-b")
            sessions,cohort=load_sessions(root,None,ATTENTION)
            items=metrics(sessions,ATTENTION)
            write_report(root,items,cohort,sessions,"attention")
            self.assertIn("UNIT-TEST FIXTURE ONLY",(root/"performance.md").read_text())
            self.assertIn("One token, four schedules",(root/"latency.svg").read_text())
            with (root/"measurements.csv").open() as f:
                rows=list(csv.DictReader(f))
            row=next(r for r in rows if r["case"]==PILOT[0].name and r["method"]==SCHEDULES[0])
            self.assertEqual(float(row["speedup_vs_mlx"]),1.0)
            self.assertEqual(int(row["observations"]),200)


if __name__=="__main__":
    unittest.main()
