const {
	GraphQLNonNull,
	GraphQLEnumType,
	GraphQLList,
	GraphQLString,
	GraphQLBoolean,
	GraphQLInputObjectType,
} = require('graphql');
const IdScalar = require('../scalars/id.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');
const CodeScalar = require('../scalars/code.scalar.js');
const CanonicalScalar = require('../scalars/canonical.scalar.js');
const DateTimeScalar = require('../scalars/datetime.scalar.js');

/**
 * @name exports
 * @summary SearchParameter Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'SearchParameter_Input',
	description:
		'A search parameter that defines a named search item that can be used to search/filter on a resource.',
	fields: () => ({
		resourceType: {
			type: new GraphQLNonNull(
				new GraphQLEnumType({
					name: 'SearchParameter_Enum_input',
					values: { SearchParameter: { value: 'SearchParameter' } },
				}),
			),
			description: 'Type of resource',
		},
		_id: {
			type: require('./element.input.js'),
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		id: {
			type: IdScalar,
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		meta: {
			type: require('./meta.input.js'),
			description:
				'The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.',
		},
		_implicitRules: {
			type: require('./element.input.js'),
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		implicitRules: {
			type: UriScalar,
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		_language: {
			type: require('./element.input.js'),
			description: 'The base language in which the resource is written.',
		},
		language: {
			type: CodeScalar,
			description: 'The base language in which the resource is written.',
		},
		text: {
			type: require('./narrative.input.js'),
			description:
				"A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it 'clinically safe' for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.",
		},
		contained: {
			type: new GraphQLList(GraphQLString),
			description:
				'These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, and nor can they have their own independent transaction scope.',
		},
		extension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		_url: {
			type: require('./element.input.js'),
			description:
				'An absolute URI that is used to identify this search parameter when it is referenced in a specification, model, design or an instance; also called its canonical identifier. This SHOULD be globally unique and SHOULD be a literal address at which at which an authoritative instance of this search parameter is (or will be) published. This URL can be the target of a canonical reference. It SHALL remain the same when the search parameter is stored on different servers.',
		},
		url: {
			type: new GraphQLNonNull(UriScalar),
			description:
				'An absolute URI that is used to identify this search parameter when it is referenced in a specification, model, design or an instance; also called its canonical identifier. This SHOULD be globally unique and SHOULD be a literal address at which at which an authoritative instance of this search parameter is (or will be) published. This URL can be the target of a canonical reference. It SHALL remain the same when the search parameter is stored on different servers.',
		},
		_version: {
			type: require('./element.input.js'),
			description:
				'The identifier that is used to identify this version of the search parameter when it is referenced in a specification, model, design or instance. This is an arbitrary value managed by the search parameter author and is not expected to be globally unique. For example, it might be a timestamp (e.g. yyyymmdd) if a managed version is not available. There is also no expectation that versions can be placed in a lexicographical sequence.',
		},
		version: {
			type: GraphQLString,
			description:
				'The identifier that is used to identify this version of the search parameter when it is referenced in a specification, model, design or instance. This is an arbitrary value managed by the search parameter author and is not expected to be globally unique. For example, it might be a timestamp (e.g. yyyymmdd) if a managed version is not available. There is also no expectation that versions can be placed in a lexicographical sequence.',
		},
		_name: {
			type: require('./element.input.js'),
			description:
				'A natural language name identifying the search parameter. This name should be usable as an identifier for the module by machine processing applications such as code generation.',
		},
		name: {
			type: new GraphQLNonNull(GraphQLString),
			description:
				'A natural language name identifying the search parameter. This name should be usable as an identifier for the module by machine processing applications such as code generation.',
		},
		_derivedFrom: {
			type: require('./element.input.js'),
			description:
				'Where this search parameter is originally defined. If a derivedFrom is provided, then the details in the search parameter must be consistent with the definition from which it is defined. i.e. the parameter should have the same meaning, and (usually) the functionality should be a proper subset of the underlying search parameter.',
		},
		derivedFrom: {
			type: CanonicalScalar,
			description:
				'Where this search parameter is originally defined. If a derivedFrom is provided, then the details in the search parameter must be consistent with the definition from which it is defined. i.e. the parameter should have the same meaning, and (usually) the functionality should be a proper subset of the underlying search parameter.',
		},
		_status: {
			type: require('./element.input.js'),
			description:
				'The status of this search parameter. Enables tracking the life-cycle of the content.',
		},
		status: {
			type: new GraphQLNonNull(CodeScalar),
			description:
				'The status of this search parameter. Enables tracking the life-cycle of the content.',
		},
		_experimental: {
			type: require('./element.input.js'),
			description:
				'A Boolean value to indicate that this search parameter is authored for testing purposes (or education/evaluation/marketing) and is not intended to be used for genuine usage.',
		},
		experimental: {
			type: GraphQLBoolean,
			description:
				'A Boolean value to indicate that this search parameter is authored for testing purposes (or education/evaluation/marketing) and is not intended to be used for genuine usage.',
		},
		_date: {
			type: require('./element.input.js'),
			description:
				'The date  (and optionally time) when the search parameter was published. The date must change when the business version changes and it must change if the status code changes. In addition, it should change when the substantive content of the search parameter changes.',
		},
		date: {
			type: DateTimeScalar,
			description:
				'The date  (and optionally time) when the search parameter was published. The date must change when the business version changes and it must change if the status code changes. In addition, it should change when the substantive content of the search parameter changes.',
		},
		_publisher: {
			type: require('./element.input.js'),
			description:
				'The name of the organization or individual that published the search parameter.',
		},
		publisher: {
			type: GraphQLString,
			description:
				'The name of the organization or individual that published the search parameter.',
		},
		contact: {
			type: new GraphQLList(require('./contactdetail.input.js')),
			description:
				'Contact details to assist a user in finding and communicating with the publisher.',
		},
		_description: {
			type: require('./element.input.js'),
			description: 'And how it used.',
		},
		description: {
			type: new GraphQLNonNull(GraphQLString),
			description: 'And how it used.',
		},
		useContext: {
			type: new GraphQLList(require('./usagecontext.input.js')),
			description:
				'The content was developed with a focus and intent of supporting the contexts that are listed. These contexts may be general categories (gender, age, ...) or may be references to specific programs (insurance plans, studies, ...) and may be used to assist with indexing and searching for appropriate search parameter instances.',
		},
		jurisdiction: {
			type: new GraphQLList(require('./codeableconcept.input.js')),
			description:
				'A legal or geographic region in which the search parameter is intended to be used.',
		},
		_purpose: {
			type: require('./element.input.js'),
			description:
				'Explanation of why this search parameter is needed and why it has been designed as it has.',
		},
		purpose: {
			type: GraphQLString,
			description:
				'Explanation of why this search parameter is needed and why it has been designed as it has.',
		},
		_code: {
			type: require('./element.input.js'),
			description:
				'The code used in the URL or the parameter name in a parameters resource for this search parameter.',
		},
		code: {
			type: new GraphQLNonNull(CodeScalar),
			description:
				'The code used in the URL or the parameter name in a parameters resource for this search parameter.',
		},
		_base: {
			type: require('./element.input.js'),
			description:
				'The base resource type(s) that this search parameter can be used against.',
		},
		base: {
			type: new GraphQLList(new GraphQLNonNull(CodeScalar)),
			description:
				'The base resource type(s) that this search parameter can be used against.',
		},
		_type: {
			type: require('./element.input.js'),
			description:
				'The type of value that a search parameter may contain, and how the content is interpreted.',
		},
		type: {
			type: new GraphQLNonNull(CodeScalar),
			description:
				'The type of value that a search parameter may contain, and how the content is interpreted.',
		},
		_expression: {
			type: require('./element.input.js'),
			description:
				'A FHIRPath expression that returns a set of elements for the search parameter.',
		},
		expression: {
			type: GraphQLString,
			description:
				'A FHIRPath expression that returns a set of elements for the search parameter.',
		},
		_xpath: {
			type: require('./element.input.js'),
			description:
				'An XPath expression that returns a set of elements for the search parameter.',
		},
		xpath: {
			type: GraphQLString,
			description:
				'An XPath expression that returns a set of elements for the search parameter.',
		},
		_xpathUsage: {
			type: require('./element.input.js'),
			description:
				'How the search parameter relates to the set of elements returned by evaluating the xpath query.',
		},
		xpathUsage: {
			type: CodeScalar,
			description:
				'How the search parameter relates to the set of elements returned by evaluating the xpath query.',
		},
		_target: {
			type: require('./element.input.js'),
			description: 'Types of resource (if a resource is referenced).',
		},
		target: {
			type: new GraphQLList(CodeScalar),
			description: 'Types of resource (if a resource is referenced).',
		},
		_multipleOr: {
			type: require('./element.input.js'),
			description:
				'Whether multiple values are allowed for each time the parameter exists. Values are separated by commas, and the parameter matches if any of the values match.',
		},
		multipleOr: {
			type: GraphQLBoolean,
			description:
				'Whether multiple values are allowed for each time the parameter exists. Values are separated by commas, and the parameter matches if any of the values match.',
		},
		_multipleAnd: {
			type: require('./element.input.js'),
			description:
				'Whether multiple parameters are allowed - e.g. more than one parameter with the same name. The search matches if all the parameters match.',
		},
		multipleAnd: {
			type: GraphQLBoolean,
			description:
				'Whether multiple parameters are allowed - e.g. more than one parameter with the same name. The search matches if all the parameters match.',
		},
		_comparator: {
			type: require('./element.input.js'),
			description: 'Comparators supported for the search parameter.',
		},
		comparator: {
			type: new GraphQLList(CodeScalar),
			description: 'Comparators supported for the search parameter.',
		},
		_modifier: {
			type: require('./element.input.js'),
			description: 'A modifier supported for the search parameter.',
		},
		modifier: {
			type: new GraphQLList(CodeScalar),
			description: 'A modifier supported for the search parameter.',
		},
		_chain: {
			type: require('./element.input.js'),
			description:
				'Contains the names of any search parameters which may be chained to the containing search parameter. Chained parameters may be added to search parameters of type reference and specify that resources will only be returned if they contain a reference to a resource which matches the chained parameter value. Values for this field should be drawn from SearchParameter.code for a parameter on the target resource type.',
		},
		chain: {
			type: new GraphQLList(GraphQLString),
			description:
				'Contains the names of any search parameters which may be chained to the containing search parameter. Chained parameters may be added to search parameters of type reference and specify that resources will only be returned if they contain a reference to a resource which matches the chained parameter value. Values for this field should be drawn from SearchParameter.code for a parameter on the target resource type.',
		},
		component: {
			type: new GraphQLList(require('./searchparametercomponent.input.js')),
			description: 'Used to define the parts of a composite search parameter.',
		},
	}),
});
