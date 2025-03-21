 
        m.def("fastgemm2one", &fastgemm2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	); 