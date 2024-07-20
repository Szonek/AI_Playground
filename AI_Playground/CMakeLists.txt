add_executable(AI_Playground
	main.cpp
	ioperator.h
	
	cuda_context.h
	cuda_context.cpp
	dx12_context.h
	dx12_context.cpp
	
	quantized_gemm.h
	quantized_gemm.cpp
	)
add_definitions(-DDML_TARGET_VERSION_USE_LATEST)
target_include_directories(AI_Playground PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AI_Playground ${NVVM_LIB} ${CUDA_LIB} d3d12 dxgi directml)

add_custom_command(TARGET AI_Playground POST_BUILD 
               COMMAND ${CMAKE_COMMAND} -E copy_if_different
			   
               "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/directml_content-src/bin/x64-win/DirectML.dll"             
               "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/directml_content-src/bin/x64-win/DirectML.Debug.dll"
			   
               "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/d3d12_content-src/build/native/bin/x64/D3D12Core.dll"             
               "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/d3d12_content-src/build/native/bin/x64/d3d12SDKLayers.dll"             
			   
			   "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/winpix/WinPixEventRuntime.dll"             
               "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/winpix/WinPixEventRuntime_UAP.dll"   
			   
                $<TARGET_FILE_DIR:AI_Playground>)
				
				
				
				