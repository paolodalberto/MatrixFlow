
        std::vector<TT> fastgemm2one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm3one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm2x2one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm3x3one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm2x3one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm3x2one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm2x2x3one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm2x3x2one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        
        std::vector<TT> fastgemm3x2x2one(int gpu,
			 std::vector<TT> a ,int lda,
			 std::vector<TT> b, int ldb,
			 std::vector<TT> c, int ldc
	);
        