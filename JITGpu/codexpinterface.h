 
        m.def("fastgemm2one", &fastgemm2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm3one", &fastgemm3one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm2x2one", &fastgemm2x2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm3x3one", &fastgemm3x3one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm2x3one", &fastgemm2x3one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm3x2one", &fastgemm3x2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm2x2x3one", &fastgemm2x2x3one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm2x3x2one", &fastgemm2x3x2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	);  
        m.def("fastgemm3x2x2one", &fastgemm3x2x2one, "GEMM",
	  py::arg("gpu"),
	  py::arg("a"), 
	  py::arg("lda"), 
	  py::arg("b"),
	  py::arg("ldb"),
	  py::arg("c"),
	  py::arg("ldc")
	); 