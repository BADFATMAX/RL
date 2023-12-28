import React, { Component, useState } from "react";
import { useEffect } from "react";

import ReactOpenApiRenderer from "@tx-dts/react-openapi-renderer";
import "@tx-dts/react-openapi-renderer/dist/index.css";

async function getJSON(){
    let r = await fetch("http://127.0.0.1:8000/openapi.json");
    let raw = await r.json();
    // console.log();
    console.log("raw: ", raw);
    return raw;
}

export default function Dataset(){
    const [spec, setSpec] = useState(null);
    useEffect(() => {
        getJSON().then((res) => setSpec(res)).then(console.log("sssss: ", spec));
      }, []);
    
    console.log("spec: ", spec);
    // const jsonSpecification = {};
    return<div>
        {spec && <ReactOpenApiRenderer specification={spec} />};
    </div>
}
