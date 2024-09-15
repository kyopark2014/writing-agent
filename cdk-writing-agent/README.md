# CDK로 인프라 설치하기 

Amazon S3를 정의하여 파일 업로드시 활용합니다. 

```python
const s3Bucket = new s3.Bucket(this, `storage-${projectName}`, {
    bucketName: bucketName,
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    autoDeleteObjects: true,
    publicReadAccess: false,
    versioned: false,
    cors: [
        {
            allowedHeaders: ['*'],
            allowedMethods: [
                s3.HttpMethods.POST,
                s3.HttpMethods.PUT,
            ],
            allowedOrigins: ['*'],
        },
    ],
});
```

대화이력을 저장하기 위해서 DynamoDB를 이용합니다. 

```python
const callLogTableName = `db-call-log-for-${projectName}`;
const callLogDataTable = new dynamodb.Table(this, `db-call-log-for-${projectName}`, {
    tableName: callLogTableName,
    partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'request_time', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const callLogIndexName = `index-type-for-${projectName}`;
callLogDataTable.addGlobalSecondaryIndex({ // GSI
    indexName: callLogIndexName,
    partitionKey: { name: 'request_id', type: dynamodb.AttributeType.STRING },
});
```

Web 화면을 위하여 CloudFront를 정의합니다. 

```python
const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
    defaultBehavior: {
        origin: new origins.S3Origin(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    },
    priceClass: cloudFront.PriceClass.PRICE_CLASS_200,
});
```

Lambda를 위한 Role을 정의합니다. 

```python
const roleLambda = new iam.Role(this, `role-lambda-chat-for-${projectName}`, {
    roleName: `role-lambda-chat-for-${projectName}-${region}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
    )
});
roleLambda.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
});
const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
    resources: ['*'],
    actions: ['bedrock:*'],
});
roleLambda.attachInlinePolicy( // add bedrock policy
    new iam.Policy(this, `bedrock-policy-lambda-chat-for-${projectName}`, {
        statements: [BedrockPolicy],
    }),
);
```

API Gateway를 생성합니다. 

```python
const role = new iam.Role(this, `api-role-for-${projectName}`, {
    roleName: `api-role-for-${projectName}-${region}`,
    assumedBy: new iam.ServicePrincipal("apigateway.amazonaws.com")
});
role.addToPolicy(new iam.PolicyStatement({
    resources: ['*'],
    actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
    ]
}));
role.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
});

const api = new apiGateway.RestApi(this, `api-chatbot-for-${projectName}`, {
    description: 'API Gateway for chatbot',
    endpointTypes: [apiGateway.EndpointType.REGIONAL],
    restApiName: 'rest-api-for-' + projectName,
    binaryMediaTypes: ['application/pdf', 'text/plain', 'text/csv'],
    deployOptions: {
        stageName: stage,

        // logging for debug
        // loggingLevel: apiGateway.MethodLoggingLevel.INFO, 
        // dataTraceEnabled: true,
    },
});

// cloudfront setting 
distribution.addBehavior("/chat", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

파일 업로드시 presigned url을 생성하는 Lambda(Upload)를 정의합니다.

```python
// Lambda - Upload
const lambdaUpload = new lambda.Function(this, `lambda-upload-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-upload-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-upload"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(10),
    environment: {
        bucketName: s3Bucket.bucketName,
        s3_prefix: s3_prefix
    }
});
s3Bucket.grantReadWrite(lambdaUpload);

// POST method - upload
const resourceName = "upload";
const upload = api.root.addResource(resourceName);
upload.addMethod('POST', new apiGateway.LambdaIntegration(lambdaUpload, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting  
distribution.addBehavior("/upload", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

DynamoDB를 조회할때 사용하는 lambda(queryResult)를 정의합니다. 

```python
// Lambda - queryResult
const lambdaQueryResult = new lambda.Function(this, `lambda-query-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-query-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-query"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName,
        indexName: callLogIndexName
    }
});
callLogDataTable.grantReadWriteData(lambdaQueryResult); // permission for dynamo

// POST method - query
const query = api.root.addResource("query");
query.addMethod('POST', new apiGateway.LambdaIntegration(lambdaQueryResult, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/query", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

WebClient에서 대화이력을 가져오기 위한 API를 위해 lambda(queryResult)를 정의합니다. 

```python
// Lambda - getHistory
const lambdaGetHistory = new lambda.Function(this, `lambda-gethistory-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-gethistory-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-gethistory"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaGetHistory); // permission for dynamo

// POST method - history
const history = api.root.addResource("history");
history.addMethod('POST', new apiGateway.LambdaIntegration(lambdaGetHistory, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/history", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

WebClient에서 대화이력을 삭제할때 필요한 lambda(delete)를 정의합니다. 

```python
// Lambda - deleteItems
const lambdaDeleteItems = new lambda.Function(this, `lambda-deleteItems-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-deleteItems-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-delete-items"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaDeleteItems); // permission for dynamo

// POST method - delete items
const deleteItem = api.root.addResource("delete");
deleteItem.addMethod('POST', new apiGateway.LambdaIntegration(lambdaDeleteItems, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/delete", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

WebSocket을 위한 API Gateway를 정의합니다. 

```python
// stream api gateway
// API Gateway
const websocketapi = new apigatewayv2.CfnApi(this, `ws-api-for-${projectName}`, {
    description: 'API Gateway for chatbot using websocket',
    apiKeySelectionExpression: "$request.header.x-api-key",
    name: 'ws-api-for-' + projectName,
    protocolType: "WEBSOCKET", // WEBSOCKET or HTTP
    routeSelectionExpression: "$request.body.action",
});
websocketapi.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); // DESTROY, RETAIN

new cdk.CfnOutput(this, 'api-identifier', {
    value: websocketapi.attrApiId,
    description: 'The API identifier.',
});

const wss_url = `wss://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
new cdk.CfnOutput(this, 'web-socket-url', {
    value: wss_url,

    description: 'The URL of Web Socket',
});

const connection_url = `https://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
new cdk.CfnOutput(this, 'connection-url', {
    value: connection_url,

    description: 'The URL of connection',
});
```

채팅을 수행하는 Lambda(chat)을 정의합니다. 

```python
// Lambda - chat (websocket)
const roleLambdaWebsocket = new iam.Role(this, `role-lambda-chat-ws-for-${projectName}`, {
    roleName: `role-lambda-chat-ws-for-${projectName}-${region}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
    )
});
roleLambdaWebsocket.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
});
roleLambdaWebsocket.attachInlinePolicy( // add bedrock policy
    new iam.Policy(this, `bedrock-policy-lambda-chat-ws-for-${projectName}`, {
        statements: [BedrockPolicy],
    }),
);
const lambdaInvokePolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: [
        "lambda:InvokeFunction"
    ],
});
roleLambdaWebsocket.attachInlinePolicy(
    new iam.Policy(this, `lambda-invoke-policy-for-${projectName}`, {
        statements: [lambdaInvokePolicy],
    }),
);

const apiInvokePolicy = new iam.PolicyStatement({
    // resources: ['arn:aws:execute-api:*:*:*'],
    resources: ['*'],
    actions: [
        'execute-api:Invoke',
        'execute-api:ManageConnections'
    ],
});
roleLambdaWebsocket.attachInlinePolicy(
    new iam.Policy(this, `api-invoke-policy-for-${projectName}`, {
        statements: [apiInvokePolicy],
    }),
);
```

외부 API를 활용할 때 필요한 암호를 저장하기 위해 secret을 정의합니다. 

```python
const langsmithApiSecret = new secretsmanager.Secret(this, `weather-langsmith-secret-for-${projectName}`, {
    description: 'secret for lamgsmith api key', // openweathermap
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    secretName: `langsmithapikey-${projectName}`,
    secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
    },
});
langsmithApiSecret.grantRead(roleLambdaWebsocket)

const tavilyApiSecret = new secretsmanager.Secret(this, `weather-tavily-secret-for-${projectName}`, {
    description: 'secret for lamgsmith api key', // openweathermap
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    secretName: `tavilyapikey-${projectName}`,
    secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
    },
});
tavilyApiSecret.grantRead(roleLambdaWebsocket)

```python
const lambdaChatWebsocket = new lambda.DockerImageFunction(this, `lambda-chat-ws-for-${projectName}`, {
    description: 'lambda for chat using websocket',
    functionName: `lambda-chat-ws-for-${projectName}`,
    code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-chat-ws')),
    timeout: cdk.Duration.seconds(900),
    memorySize: 8192,
    role: roleLambdaWebsocket,
    environment: {
        s3_bucket: s3Bucket.bucketName,
        s3_prefix: s3_prefix,
        path: 'https://' + distribution.domainName + '/',
        callLogTableName: callLogTableName,
        LLM_for_chat: JSON.stringify(LLM_for_chat),
        LLM_for_multimodal: JSON.stringify(LLM_for_multimodal),
        LLM_embedding: JSON.stringify(titan_embedding_v2),
        connection_url: connection_url,
        debugMessageMode: debugMessageMode,
        projectName: projectName,
        prompt_flow_name: prompt_flow_name,
        rag_prompt_flow_name: rag_prompt_flow_name,
        knowledge_base_name: knowledge_base_name
    }
});
lambdaChatWebsocket.grantInvoke(new iam.ServicePrincipal('apigateway.amazonaws.com'));
s3Bucket.grantReadWrite(lambdaChatWebsocket); // permission for s3
callLogDataTable.grantReadWriteData(lambdaChatWebsocket); // permission for dynamo 
```

WebSocket을 위한 API Gateway의 리소스를 정의합니다. 

```python
const integrationUri = `arn:aws:apigateway:${region}:lambda:path/2015-03-31/functions/${lambdaChatWebsocket.functionArn}/invocations`;
const cfnIntegration = new apigatewayv2.CfnIntegration(this, `api-integration-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    integrationType: 'AWS_PROXY',
    credentialsArn: role.roleArn,
    connectionType: 'INTERNET',
    description: 'Integration for connect',
    integrationUri: integrationUri,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-connect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$connect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'connect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-disconnect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$disconnect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'disconnect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-default`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$default",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'default',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnStage(this, `api-stage-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    stageName: stage
});
```

WebSocket이 접속할 endpoint를 WebClient에게 알려주기 위한 API를 정의하기 위하여 lambda(provisioning)을 정의합니다. 

```python
// lambda - provisioning
const lambdaProvisioning = new lambda.Function(this, `lambda-provisioning-for-${projectName}`, {
    description: 'lambda to earn provisioning info',
    functionName: `lambda-provisioning-api-${projectName}`,
    handler: 'lambda_function.lambda_handler',
    runtime: lambda.Runtime.PYTHON_3_11,
    code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-provisioning')),
    timeout: cdk.Duration.seconds(30),
    environment: {
        wss_url: wss_url,
    }
});

// POST method - provisioning
const provisioning_info = api.root.addResource("provisioning");
provisioning_info.addMethod('POST', new apiGateway.LambdaIntegration(lambdaProvisioning, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for provisioning api
distribution.addBehavior("/provisioning", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

API Gateway를 멀티스택으로 배포하기 위하여 component  deployment를 분리하였습니다.

```python
export class componentDeployment extends cdk.Stack {
    constructor(scope: Construct, id: string, appId: string, props?: cdk.StackProps) {
        super(scope, id, props);

        new apigatewayv2.CfnDeployment(this, `api-deployment-for-${projectName}`, {
            apiId: appId,
            description: "deploy api gateway using websocker",  // $default
            stageName: stage
        });
    }
```
